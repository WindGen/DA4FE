"""Generate Stage-3 images and keep the most similar real/generated pairs.

The script loads a Stage-3 generation checkpoint, generates the requested
number of candidates for every sample in the train/validation/test splits,
scores each candidate against its real image with CLIP image embeddings, and
keeps at most one candidate for each original image.  Only the global top-K
pairs are copied to the final output directory; all candidate scores are
written to ``similarities.csv``.

Example::

    python scripts/select_top_generation_pairs.py \
        --checkpoint ../result/generation/20261001_120000/checkpoint.pth \
        --output-dir ../result/generation/20261001_120000/top100_selection \
        --num-samples 3 --top-k 100 --batch-size 1
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.exp_stage3_generation import Exp_Stage3_Generation  # noqa: E402


DEFAULTS = {
    "stage1_run_dir": None,
    "stage1_checkpoint": None,
    "stage1_run_params": None,
    "stage1_split_summary": None,
    "eeg_hf_dataset_id": None,
    "eeg_hf_cache_dir": None,
    "eeg_hf_split_mode": None,
    "eeg_hf_split_seed": None,
    "eeg_hf_train_ratio": None,
    "eeg_hf_val_ratio": None,
    "eeg_hf_test_ratio": None,
    "pretrained_root": "../pretrained",
    "sd_model_path": None,
    "clip_model_path": None,
    "hf_cache_dir": None,
    "allow_hf_download": True,
    "generation_context_tokens": 4,
    "generation_condition_dim": 768,
    "generation_adapter_hidden_dim": 1024,
    "generation_adapter_dropout": 0.0,
    "generation_image_size": 512,
    "generation_epochs": 1,
    "generation_batch_size": 1,
    "generation_lr": 1e-4,
    "generation_weight_decay": 1e-2,
    "generation_clip_weight": 0.0,
    "generation_stage1_lr": 0.0,
    "generation_train_unet": False,
    "generation_grad_clip": 1.0,
    "generation_num_inference_steps": 50,
    "generation_num_samples": 1,
    "generation_eval_items": 0,
    "generation_skip_sampling": True,
    "output_dir": None,
    "num_workers": 0,
    "seed": 42,
    "use_gpu": True,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def str_or_none(value):
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def resolve_checkpoint(value):
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "checkpoint.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Stage-3 checkpoint does not exist: {path}")
    return path.resolve()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("args"), dict):
        return dict(payload["args"])
    return dict(payload) if isinstance(payload, dict) else {}


def load_run_args(run_params_path):
    values = dict(DEFAULTS)
    if run_params_path is not None and Path(run_params_path).is_file():
        values.update(load_json(run_params_path))
    return values


def resolve_relative(value, base_dir):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = base_dir / path
    return str(candidate) if candidate.exists() else str(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage3_state(experiment, checkpoint):
    adapter_state = checkpoint.get("condition_adapter") if isinstance(checkpoint, dict) else None
    if adapter_state is None:
        raise KeyError("The checkpoint does not contain 'condition_adapter'. Is this a Stage-3 checkpoint?")
    experiment.condition_adapter.load_state_dict(adapter_state)

    unet_state = checkpoint.get("unet_trainable") if isinstance(checkpoint, dict) else None
    if unet_state:
        experiment.unet.load_state_dict(unet_state, strict=False)

    stage1_state = checkpoint.get("stage1_model") if isinstance(checkpoint, dict) else None
    if stage1_state is not None:
        experiment.stage1_model.load_state_dict(stage1_state, strict=False)

    experiment.stage1_model.eval()
    experiment.condition_adapter.eval()
    experiment.unet.eval()
    experiment.vae.eval()


def pil_batch_to_tensor(images):
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0 for image in images]
    return torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()


def tensor_to_uint8(image):
    return ((image.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte().permute(1, 2, 0).numpy()


def image_hash(image):
    return hashlib.sha1(tensor_to_uint8(image).tobytes()).hexdigest()


def safe_component(value):
    text = str(value).strip() or "unknown"
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in text)


def clip_values(images):
    images = (images + 1.0) / 2.0
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (images - mean) / std


def clip_embeddings(model, images, device):
    values = clip_values(images.to(device=device, dtype=torch.float32))
    embeddings = model(pixel_values=values).image_embeds.float()
    return F.normalize(embeddings, dim=-1)


def sample_from_context(experiment, context):
    batch_size = context.shape[0]
    latents = torch.randn(
        batch_size,
        experiment.unet.config.in_channels,
        experiment.args.generation_image_size // 8,
        experiment.args.generation_image_size // 8,
        device=experiment.device,
    )
    experiment.scheduler.set_timesteps(experiment.args.generation_num_inference_steps, device=experiment.device)
    for timestep in experiment.scheduler.timesteps:
        model_input = experiment.scheduler.scale_model_input(latents, timestep)
        noise_pred = experiment.unet(model_input, timestep, encoder_hidden_states=context).sample
        latents = experiment.scheduler.step(noise_pred, timestep, latents).prev_sample
    return experiment._decode(latents)


def build_parser():
    parser = argparse.ArgumentParser(description="Select the top real/generated image pairs from a Stage-3 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Stage-3 checkpoint.pth or its containing run directory")
    parser.add_argument("--run-params", type=str_or_none, default=None, help="generation run_params.json; defaults to checkpoint directory")
    parser.add_argument("--stage1-run-dir", type=str_or_none, default=None, help="override the DA4FE Stage-1 run directory")
    parser.add_argument("--stage1-checkpoint", type=str_or_none, default=None, help="override the DA4FE Stage-1 checkpoint")
    parser.add_argument("--sd-model-path", type=str_or_none, default=None, help="override Stable Diffusion model path or Hugging Face id")
    parser.add_argument("--clip-model-path", type=str_or_none, default=None, help="override CLIP image model path or Hugging Face id")
    parser.add_argument("--hf-cache-dir", type=str_or_none, default=None, help="Hugging Face cache directory; none uses the default")
    parser.add_argument("--allow-hf-download", type=str2bool, default=None, help="allow missing Hugging Face models to download")
    parser.add_argument("--output-dir", type=str_or_none, default=None, help="output directory; defaults to checkpoint/top100_selection")
    parser.add_argument("--top-k", type=int, default=100, help="number of pairs to retain globally")
    parser.add_argument("--num-samples", type=int, default=None, help="generated candidates per EEG; defaults to the training setting")
    parser.add_argument("--inference-steps", type=int, default=None, help="diffusion inference steps; defaults to the training setting")
    parser.add_argument("--batch-size", type=int, default=1, help="inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count; zero is safest on virtual GPUs")
    parser.add_argument("--generation-image-size", type=int, default=None, help="override generation image size; normally keep the training value")
    parser.add_argument("--use-gpu", type=str2bool, default=None, help="override the saved GPU setting")
    parser.add_argument("--seed", type=int, default=None, help="random seed; defaults to the saved training seed")
    parser.add_argument("--group-by", choices=["image", "sample"], default="image", help="deduplicate by exact real-image pixels or by split/sample")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"], help="splits to process")
    return parser


def main():
    cli = build_parser().parse_args()
    if cli.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if cli.num_samples is not None and cli.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    checkpoint_path = resolve_checkpoint(cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The supplied checkpoint is not a Stage-3 checkpoint dictionary")

    run_params_path = Path(cli.run_params).expanduser() if cli.run_params else checkpoint_path.parent / "run_params.json"
    values = load_run_args(run_params_path)
    values["output_dir"] = str(Path(cli.output_dir).expanduser() if cli.output_dir else checkpoint_path.parent / "top100_selection")
    values["generation_clip_weight"] = 0.0
    values["generation_stage1_lr"] = 0.0
    values["generation_batch_size"] = int(cli.batch_size)
    values["num_workers"] = int(cli.num_workers)
    if cli.num_samples is not None:
        values["generation_num_samples"] = int(cli.num_samples)
    if cli.inference_steps is not None:
        values["generation_num_inference_steps"] = int(cli.inference_steps)
    if cli.generation_image_size is not None:
        values["generation_image_size"] = int(cli.generation_image_size)
    if cli.stage1_run_dir is not None:
        values["stage1_run_dir"] = cli.stage1_run_dir
    if cli.stage1_checkpoint is not None:
        values["stage1_checkpoint"] = cli.stage1_checkpoint
    if cli.sd_model_path is not None:
        values["sd_model_path"] = cli.sd_model_path
    if cli.clip_model_path is not None:
        values["clip_model_path"] = cli.clip_model_path
    if cli.hf_cache_dir is not None:
        values["hf_cache_dir"] = cli.hf_cache_dir
    if cli.allow_hf_download is not None:
        values["allow_hf_download"] = cli.allow_hf_download
    if cli.use_gpu is not None:
        values["use_gpu"] = cli.use_gpu
    if cli.seed is not None:
        values["seed"] = cli.seed

    saved_stage1_checkpoint = checkpoint.get("stage1_checkpoint")
    if values.get("stage1_checkpoint") is None and saved_stage1_checkpoint:
        values["stage1_checkpoint"] = resolve_relative(saved_stage1_checkpoint, checkpoint_path.parent)
    if values.get("stage1_run_dir"):
        values["stage1_run_dir"] = resolve_relative(values["stage1_run_dir"], ROOT)
    if values.get("stage1_checkpoint"):
        values["stage1_checkpoint"] = resolve_relative(values["stage1_checkpoint"], ROOT)

    args = SimpleNamespace(**values)
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    print(f"Loading Stage-3 checkpoint: {checkpoint_path}")
    print(f"Using output directory: {output_dir}")

    experiment = Exp_Stage3_Generation(args)
    load_stage3_state(experiment, checkpoint)

    try:
        from transformers import CLIPVisionModelWithProjection
    except ImportError as exc:
        raise ImportError("This script requires transformers>=4.40.0 for CLIP similarity") from exc
    clip_path = experiment._resolve_pretrained_path(args.clip_model_path, "clip-vit-large-patch14")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        str(clip_path),
        cache_dir=args.hf_cache_dir,
        local_files_only=not args.allow_hf_download,
    ).to(experiment.device).eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    temp_root = output_dir / ".candidate_cache"
    temp_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    best_by_source = {}
    total_candidates = 0
    try:
        with torch.inference_mode():
            for split in cli.splits:
                dataset = getattr(experiment, f"{split}_data")
                # Training datasets normally use random crops.  Similarity
                # selection must compare against a stable real image.
                dataset.training = False
                loader = experiment._loader(dataset, train=False)
                for dataset_index, batch in enumerate(tqdm(loader, desc=f"Generating {split}")):
                    eeg = batch["eeg"].to(experiment.device, non_blocking=True).float()
                    real_images = batch["image"].float()
                    real_embeddings = clip_embeddings(clip_model, real_images, experiment.device)
                    features = experiment._features(eeg)
                    context = experiment.condition_adapter(features)
                    real_hashes = [image_hash(image) for image in real_images]

                    for candidate_index in range(int(args.generation_num_samples)):
                        generated_images = sample_from_context(experiment, context)
                        generated_tensors = pil_batch_to_tensor(generated_images)
                        generated_embeddings = clip_embeddings(clip_model, generated_tensors, experiment.device)
                        scores = (real_embeddings * generated_embeddings).sum(dim=-1).detach().cpu().tolist()

                        for item_index, score in enumerate(scores):
                            sample_number = dataset_index * int(args.generation_batch_size) + item_index
                            source_key = real_hashes[item_index] if cli.group_by == "image" else f"{split}:{sample_number}"
                            source_file_key = hashlib.sha1(source_key.encode("utf-8")).hexdigest()
                            row = {
                                "split": split,
                                "sample_index": sample_number,
                                "candidate_index": candidate_index + 1,
                                "source_key": source_key,
                                "real_image_hash": real_hashes[item_index],
                                "subject": str(batch["subject"][item_index]),
                                "label": str(batch["label_name"][item_index]),
                                "similarity": float(score),
                                "best_for_source": False,
                                "top_k_rank": "",
                                "real_image_path": "",
                                "generated_image_path": "",
                            }
                            all_rows.append(row)
                            total_candidates += 1
                            previous = best_by_source.get(source_key)
                            if previous is None or float(score) > float(previous["similarity"]):
                                if previous is not None:
                                    previous["best_for_source"] = False
                                row["best_for_source"] = True
                                real_path = temp_root / f"{source_file_key}_real.png"
                                generated_path = temp_root / f"{source_file_key}_generated.png"
                                if not real_path.exists():
                                    Image.fromarray(tensor_to_uint8(real_images[item_index])).save(real_path)
                                generated_images[item_index].save(generated_path)
                                row["_real_temp"] = str(real_path)
                                row["_generated_temp"] = str(generated_path)
                                best_by_source[source_key] = row

        ranked = sorted(best_by_source.values(), key=lambda item: float(item["similarity"]), reverse=True)
        selected = ranked[: int(cli.top_k)]
        pairs_dir = output_dir / "top100_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        top_rows = []
        selected_ids = set()
        for rank, row in enumerate(selected, start=1):
            stem = f"{rank:03d}_{safe_component(row['split'])}_{int(row['sample_index']):07d}"
            real_path = pairs_dir / f"{stem}_real.png"
            generated_path = pairs_dir / f"{stem}_generated.png"
            shutil.copy2(row["_real_temp"], real_path)
            shutil.copy2(row["_generated_temp"], generated_path)
            row["top_k_rank"] = rank
            row["real_image_path"] = str(real_path.resolve())
            row["generated_image_path"] = str(generated_path.resolve())
            selected_ids.add(id(row))
            top_rows.append({key: value for key, value in row.items() if not key.startswith("_")})

        similarity_fields = [
            "split", "sample_index", "candidate_index", "source_key", "real_image_hash",
            "subject", "label", "similarity", "best_for_source", "top_k_rank",
            "real_image_path", "generated_image_path",
        ]
        for row in all_rows:
            if id(row) not in selected_ids:
                row["top_k_rank"] = ""
                row["real_image_path"] = ""
                row["generated_image_path"] = ""
        with (output_dir / "similarities.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=similarity_fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in similarity_fields} for row in all_rows)
        with (output_dir / "top100_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=similarity_fields)
            writer.writeheader()
            writer.writerows(top_rows)

        summary = {
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "splits": cli.splits,
            "group_by": cli.group_by,
            "top_k": int(cli.top_k),
            "num_samples_per_eeg": int(args.generation_num_samples),
            "inference_steps": int(args.generation_num_inference_steps),
            "total_candidates": total_candidates,
            "unique_sources": len(best_by_source),
            "retained_pairs": len(selected),
            "clip_model": str(clip_path),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"Scored {total_candidates} generated candidates from {len(best_by_source)} source images/samples.")
        print(f"Saved top {len(selected)} pairs to {pairs_dir}")
        print(f"Saved all similarities to {output_dir / 'similarities.csv'}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
