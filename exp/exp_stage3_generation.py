"""DreamDiffusion-style image generation conditioned on DA4FE EEG features."""

import csv
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_provider.generation_dataset import EEGImageNetHuggingFaceGenerationDataset, generation_collate
from models.eeg_sd_condition import EEGConditionAdapter
from utils.checkpointing import load_checkpoint_object, load_model_state, unwrap_model
from exp.exp_stage1_feature import Stage1FeatureModel
from models import DA4FE


def _as_namespace(payload):
    return SimpleNamespace(**(payload.get("args", payload) if isinstance(payload, dict) else {}))


def _first_existing(paths):
    for path in paths:
        if path is not None and Path(path).is_file():
            return Path(path)
    return None


class Exp_Stage3_Generation:
    """Train an EEG condition adapter and optionally SD cross-attention layers."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.stage1_args, self.stage1_checkpoint_path = self._load_stage1_metadata()
        self._apply_stage1_overrides()
        self.train_data = EEGImageNetHuggingFaceGenerationDataset(self._dataset_args(), "TRAIN")
        self.val_data = EEGImageNetHuggingFaceGenerationDataset(self._dataset_args(), "VAL")
        self.test_data = EEGImageNetHuggingFaceGenerationDataset(self._dataset_args(), "TEST")
        self.stage1_model = self._build_stage1_model().to(self.device)
        self.feature_dim = int(self.stage1_model.feature_dim)
        self._build_diffusion_models()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_stage1_metadata(self):
        run_dir = Path(self.args.stage1_run_dir).expanduser() if self.args.stage1_run_dir else None
        if run_dir is not None and run_dir.is_file():
            run_dir = run_dir.parent
        if run_dir is None and self.args.stage1_checkpoint:
            checkpoint_hint = Path(self.args.stage1_checkpoint).expanduser()
            run_dir = checkpoint_hint.parent if checkpoint_hint.is_file() else None
        if run_dir is None and self.args.stage1_run_params:
            run_params_hint = Path(self.args.stage1_run_params).expanduser()
            run_dir = run_params_hint.parent if run_params_hint.is_file() else None
        run_params_path = _first_existing([self.args.stage1_run_params, run_dir / "run_params.json" if run_dir else None])
        split_summary_path = _first_existing([self.args.stage1_split_summary, run_dir / "split_summary.json" if run_dir else None])
        if run_params_path is None:
            raise FileNotFoundError("Could not find run_params.json; pass --stage1_run_dir or --stage1_run_params")
        with run_params_path.open("r", encoding="utf-8") as handle:
            stage1_args = _as_namespace(json.load(handle))
        self.stage1_split_summary = None
        if split_summary_path is not None:
            with split_summary_path.open("r", encoding="utf-8") as handle:
                self.stage1_split_summary = json.load(handle)
        checkpoint_path = _first_existing([self.args.stage1_checkpoint, run_dir / "checkpoint.pth" if run_dir else None])
        if checkpoint_path is None and run_dir is not None:
            candidates = sorted(run_dir.rglob("*.pth"))
            checkpoint_path = candidates[0] if candidates else None
        if checkpoint_path is None:
            raise FileNotFoundError("Could not find checkpoint.pth; pass --stage1_checkpoint")
        self.stage1_run_params_path = run_params_path
        self.stage1_split_summary_path = split_summary_path
        return stage1_args, checkpoint_path

    def _apply_stage1_overrides(self):
        """Use CLI values when supplied, otherwise retain DA4FE run parameters."""
        for name in ("eeg_hf_dataset_id", "eeg_hf_cache_dir", "eeg_hf_split_mode", "eeg_hf_split_seed", "eeg_hf_train_ratio", "eeg_hf_val_ratio", "eeg_hf_test_ratio"):
            value = getattr(self.args, name, None)
            if value is not None:
                setattr(self.stage1_args, name, value)
        self.stage1_args.requested_seq_len = getattr(self.stage1_args, "requested_seq_len", getattr(self.stage1_args, "seq_len", 512))
        self.stage1_args.seq_len = int(getattr(self.stage1_args, "seq_len", self.stage1_args.requested_seq_len))
        self.stage1_args.eeg_adaptive_seq_len = getattr(self.stage1_args, "eeg_adaptive_seq_len", True)
        self.stage1_args.eeg_normalize = getattr(self.stage1_args, "eeg_normalize", True)
        self.stage1_args.seed = int(getattr(self.stage1_args, "seed", self.args.seed))

    def _dataset_args(self):
        dataset_args = SimpleNamespace(**vars(self.stage1_args))
        dataset_args.generation_image_size = self.args.generation_image_size
        dataset_args.eeg_hf_split_mode = getattr(self.stage1_args, "eeg_hf_split_mode", "original")
        dataset_args.eeg_hf_split_seed = getattr(self.stage1_args, "eeg_hf_split_seed", self.args.seed)
        dataset_args.num_workers = self.args.num_workers
        return dataset_args

    def _build_stage1_model(self):
        stage1_args = self.stage1_args
        stage1_args.model = getattr(stage1_args, "model", "DA4FE")
        if stage1_args.model != "DA4FE":
            raise ValueError(f"Stage-3 currently expects a DA4FE checkpoint, got {stage1_args.model!r}")
        stage1_args.num_class = int(getattr(stage1_args, "num_class", 0) or self.train_data.num_class)
        stage1_args.enc_in = int(getattr(stage1_args, "enc_in", self.train_data.feature_dim))
        stage1_args.use_multi_gpu = False
        stage1_args.stage1_l2_normalize = getattr(stage1_args, "stage1_l2_normalize", True)
        backbone = DA4FE.Model(stage1_args).float()
        feature_dim = int(getattr(stage1_args, "stage1_backbone_feature_dim", getattr(stage1_args, "da4fe_fusion_out_dim", 512)))
        model = Stage1FeatureModel(backbone=backbone, feature_dim=feature_dim, num_class=stage1_args.num_class, args=stage1_args).float()
        checkpoint = load_checkpoint_object(self.stage1_checkpoint_path, map_location="cpu")
        try:
            load_model_state(model, checkpoint)
        except RuntimeError:
            if isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
                state = checkpoint["backbone_state_dict"]
                state = {key.removeprefix("module."): value for key, value in state.items()}
                model.backbone.load_state_dict(state)
            else:
                raise
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = self.args.generation_stage1_lr > 0
        return model

    def _resolve_pretrained_path(self, explicit, default_name):
        path = Path(explicit).expanduser() if explicit else Path(self.args.pretrained_root).expanduser() / default_name
        if explicit is None and not path.exists() and default_name.startswith("stable-diffusion"):
            root = Path(self.args.pretrained_root).expanduser()
            candidates = sorted(list(root.glob("*.safetensors")) + list(root.glob("*.ckpt"))) if root.is_dir() else []
            if len(candidates) == 1:
                path = candidates[0]
        if path.exists():
            return path
        if self.args.allow_hf_download:
            if explicit is not None and "/" in str(explicit).replace("\\", "/"):
                return str(explicit)
            return "runwayml/stable-diffusion-v1-5" if default_name.startswith("stable-diffusion") else "openai/clip-vit-large-patch14"
        raise FileNotFoundError(f"Pretrained model path does not exist: {path}. Set --allow_hf_download true to download from Hugging Face.")

    def _build_diffusion_models(self):
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError("Stage-3 requires diffusers>=0.27.0. Install requirements.txt before training.") from exc
        sd_path = self._resolve_pretrained_path(self.args.sd_model_path, "stable-diffusion-v1-5")
        if isinstance(sd_path, str):
            pipeline = StableDiffusionPipeline.from_pretrained(str(sd_path), cache_dir=self.args.hf_cache_dir, local_files_only=not self.args.allow_hf_download, safety_checker=None)
        elif sd_path.is_file() and hasattr(StableDiffusionPipeline, "from_single_file"):
            pipeline = StableDiffusionPipeline.from_single_file(str(sd_path), local_files_only=True, safety_checker=None)
        elif sd_path.is_dir():
            pipeline = StableDiffusionPipeline.from_pretrained(str(sd_path), local_files_only=True, safety_checker=None)
        else:
            raise ValueError(f"Unsupported Stable Diffusion weight path: {sd_path}")
        self.vae = pipeline.vae.to(self.device).eval()
        self.unet = pipeline.unet.to(self.device)
        self.scheduler = pipeline.scheduler
        del pipeline
        for parameter in self.vae.parameters():
            parameter.requires_grad = False
        for parameter in self.unet.parameters():
            parameter.requires_grad = False
        cross_dim = getattr(self.unet.config, "cross_attention_dim", self.args.generation_condition_dim)
        if isinstance(cross_dim, (tuple, list)):
            cross_dim = int(cross_dim[0])
        if int(cross_dim) != int(self.args.generation_condition_dim):
            raise ValueError(f"generation_condition_dim={self.args.generation_condition_dim} does not match U-Net cross_attention_dim={cross_dim}")
        self.condition_adapter = EEGConditionAdapter(self.feature_dim, context_dim=int(cross_dim), num_tokens=self.args.generation_context_tokens, hidden_dim=self.args.generation_adapter_hidden_dim, dropout=self.args.generation_adapter_dropout).to(self.device)
        self.trainable_unet_parameters = []
        if self.args.generation_train_unet:
            for name, parameter in self.unet.named_parameters():
                if "attn2" in name or "cross" in name:
                    parameter.requires_grad = True
                    self.trainable_unet_parameters.append(parameter)
        self.clip_model = None
        if self.args.generation_clip_weight > 0:
            try:
                from transformers import CLIPVisionModelWithProjection
            except ImportError as exc:
                raise ImportError("CLIP loss requires transformers>=4.40.0") from exc
            clip_path = self._resolve_pretrained_path(self.args.clip_model_path, "clip-vit-large-patch14")
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(str(clip_path), cache_dir=self.args.hf_cache_dir, local_files_only=not self.args.allow_hf_download).to(self.device).eval()
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = False
        params = [{"params": list(self.condition_adapter.parameters()) + self.trainable_unet_parameters, "lr": self.args.generation_lr}]
        if self.args.generation_stage1_lr > 0:
            params.append({"params": [parameter for parameter in self.stage1_model.parameters() if parameter.requires_grad], "lr": self.args.generation_stage1_lr})
        self.optimizer = torch.optim.AdamW(params, lr=self.args.generation_lr, weight_decay=self.args.generation_weight_decay)

    def _loader(self, dataset, train=False):
        return DataLoader(dataset, batch_size=self.args.generation_batch_size, shuffle=train, num_workers=self.args.num_workers, drop_last=False, collate_fn=lambda batch: generation_collate(batch, max_len=self.stage1_args.seq_len))

    def _features(self, eeg):
        if self.args.generation_stage1_lr > 0:
            return self.stage1_model(eeg, return_features=True)[1]
        with torch.no_grad():
            return self.stage1_model(eeg, return_features=True)[1]

    def _clip_values(self, images):
        images = (images + 1.0) / 2.0
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        return (images - mean) / std

    def _loss_for_batch(self, batch):
        eeg = batch["eeg"].to(self.device, non_blocking=True).float()
        images = batch["image"].to(self.device, non_blocking=True).float()
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        features = self._features(eeg)
        context = self.condition_adapter(features)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=context).sample
        diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
        clip_loss = torch.zeros((), device=self.device)
        if self.clip_model is not None:
            alpha = self.scheduler.alphas_cumprod.to(self.device)[timesteps].view(-1, 1, 1, 1)
            pred_latents = (noisy_latents - (1.0 - alpha).sqrt() * noise_pred) / alpha.sqrt().clamp_min(1e-5)
            decoded = self.vae.decode(pred_latents / self.vae.config.scaling_factor).sample
            predicted_embeds = self.clip_model(pixel_values=self._clip_values(decoded)).image_embeds
            with torch.no_grad():
                target_embeds = self.clip_model(pixel_values=self._clip_values(images)).image_embeds
            clip_loss = 1.0 - F.cosine_similarity(predicted_embeds, target_embeds, dim=-1).mean()
        total = diffusion_loss + self.args.generation_clip_weight * clip_loss
        return total, diffusion_loss.detach(), clip_loss.detach()

    def _evaluate(self, dataset, split):
        self.stage1_model.eval()
        self.condition_adapter.eval()
        self.unet.eval()
        total = diffusion = clip = 0.0
        count = 0
        with torch.no_grad():
            for batch in self._loader(dataset, train=False):
                value, diffusion_value, clip_value = self._loss_for_batch(batch)
                size = batch["eeg"].shape[0]
                total += float(value) * size
                diffusion += float(diffusion_value) * size
                clip += float(clip_value) * size
                count += size
        result = {"loss": total / max(count, 1), "diffusion_loss": diffusion / max(count, 1), "clip_loss": clip / max(count, 1)}
        print(f"{split}: loss={result['loss']:.6f}, diffusion={result['diffusion_loss']:.6f}, clip={result['clip_loss']:.6f}")
        return result

    def _save_checkpoint(self, epoch, metrics):
        path = self.output_dir / "checkpoint.pth"
        trainable_names = {name for name, parameter in self.unet.named_parameters() if parameter.requires_grad}
        unet_state = {name: value.detach().cpu() for name, value in self.unet.state_dict().items() if name in trainable_names}
        stage1_state = self.stage1_model.state_dict() if self.args.generation_stage1_lr > 0 else None
        torch.save({"epoch": epoch, "metrics": metrics, "condition_adapter": self.condition_adapter.state_dict(), "unet_trainable": unet_state, "stage1_model": stage1_state, "optimizer": self.optimizer.state_dict(), "stage1_checkpoint": str(self.stage1_checkpoint_path)}, path)
        return path

    def train(self):
        best = math.inf
        history = []
        epoch_progress = tqdm(range(1, self.args.generation_epochs + 1), desc="Stage-3 total", unit="epoch", position=0, dynamic_ncols=True)
        for epoch in epoch_progress:
            self.stage1_model.train(self.args.generation_stage1_lr > 0)
            self.condition_adapter.train()
            self.unet.train(self.args.generation_train_unet)
            totals = np.zeros(3, dtype=np.float64)
            count = 0
            start = time.time()
            batch_progress = tqdm(self._loader(self.train_data, train=True), desc=f"Epoch {epoch}/{self.args.generation_epochs}", unit="batch", position=1, leave=False, dynamic_ncols=True)
            for batch in batch_progress:
                self.optimizer.zero_grad(set_to_none=True)
                value, diffusion_value, clip_value = self._loss_for_batch(batch)
                value.backward()
                torch.nn.utils.clip_grad_norm_(self.condition_adapter.parameters(), self.args.generation_grad_clip)
                self.optimizer.step()
                size = batch["eeg"].shape[0]
                totals += np.asarray([float(value), float(diffusion_value), float(clip_value)]) * size
                count += size
                batch_progress.set_postfix(loss=f"{float(value):.4f}", refresh=False)
            train_metrics = {"loss": totals[0] / max(count, 1), "diffusion_loss": totals[1] / max(count, 1), "clip_loss": totals[2] / max(count, 1)}
            val_metrics = self._evaluate(self.val_data, "val")
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "epoch_time_sec": time.time() - start}
            history.append(row)
            epoch_progress.set_postfix(train_loss=f"{train_metrics['loss']:.4f}", val_loss=f"{val_metrics['loss']:.4f}", refresh=True)
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f}, val_loss={val_metrics['loss']:.6f}")
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                self._save_checkpoint(epoch, row)
        epoch_progress.close()
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        return history

    def _decode(self, latents):
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        images = ((images.clamp(-1, 1) + 1.0) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(image) for image in images]

    @staticmethod
    def _tensor_to_pil(image):
        image = ((image.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        return Image.fromarray(image.permute(1, 2, 0).numpy())

    def _sample_batch(self, eeg):
        features = self._features(eeg)
        context = self.condition_adapter(features)
        latents = torch.randn(eeg.shape[0], self.unet.config.in_channels, self.args.generation_image_size // 8, self.args.generation_image_size // 8, device=self.device)
        self.scheduler.set_timesteps(self.args.generation_num_inference_steps, device=self.device)
        for timestep in self.scheduler.timesteps:
            model_input = self.scheduler.scale_model_input(latents, timestep)
            noise_pred = self.unet(model_input, timestep, encoder_hidden_states=context).sample
            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        return self._decode(latents)

    def sample_split(self, dataset, split):
        root = self.output_dir / f"samples_{split.lower()}"
        root.mkdir(parents=True, exist_ok=True)
        self.stage1_model.eval()
        self.condition_adapter.eval()
        self.unet.eval()
        saved_items = 0
        manifest_rows = []
        with torch.no_grad():
            for batch in tqdm(self._loader(dataset, train=False), desc=f"Sampling {split}"):
                eeg = batch["eeg"].to(self.device).float()
                for sample_index in range(self.args.generation_num_samples):
                    images = self._sample_batch(eeg)
                    for item_index, image in enumerate(images):
                        if sample_index == 0:
                            sample_id = f"{saved_items + item_index:06d}"
                            real_name = f"{sample_id}_0.png"
                            self._tensor_to_pil(batch["image"][item_index]).save(root / real_name)
                            manifest_rows.append({"sample_id": sample_id, "real_image": real_name, "subject": batch["subject"][item_index], "label": batch["label_name"][item_index]})
                        else:
                            sample_id = f"{saved_items + item_index:06d}"
                        generated_name = f"{sample_id}_{sample_index + 1}.png"
                        image.save(root / generated_name)
                        manifest_rows[saved_items + item_index][f"generated_{sample_index + 1}"] = generated_name
                    if sample_index == self.args.generation_num_samples - 1:
                        saved_items += len(images)
                if self.args.generation_eval_items > 0 and saved_items >= self.args.generation_eval_items:
                    break
        manifest_path = root / "manifest.csv"
        fieldnames = ["sample_id", "real_image", "subject", "label"] + [f"generated_{index}" for index in range(1, self.args.generation_num_samples + 1)]
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Saved {saved_items} real/generated image groups to {root}")
