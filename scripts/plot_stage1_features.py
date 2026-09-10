"""Extract Stage-1 embeddings and draw a 2-D feature distribution.

The script is intentionally driven by the ``run_params.json`` and
``split_summary.json`` files written next to a Stage-1 checkpoint.  This keeps
the model and data configuration identical to the training run, without
requiring model dimensions to be entered by hand.  The checkpoint's embedded
args remain a fallback for older runs.

Example (local EEG-ImageNet):

    python scripts/plot_stage1_features.py \
        --checkpoint checkpoints/DA4FE/stage1/<run>/checkpoint.pth \
        --method tsne

The script writes all normalized embeddings to ``features.npz`` and creates
one comparable plot plus one plot for each available split.  Runs trained with
``use_validation=false`` produce only train and test panels; the original
training and validation samples are merged into the train panel to match the
training procedure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Allow the script to be launched from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.exp_stage1_feature import Exp_Stage1_Feature  # noqa: E402
from utils.checkpointing import load_model_state  # noqa: E402


DEFAULT_ARGS: dict[str, Any] = {
    # Dataset/model defaults mirror run_stage1.py.  Values embedded in the
    # checkpoint take precedence over these defaults.
    "model": "DA4FE",
    "data": "EEG-ImageNet-HF",
    "root_path": None,
    "data_path": "EEG-ImageNet",
    "eeg_hf_dataset_id": "luigi-s/EEG_Image_CVPR_ALL_subj",
    "eeg_hf_cache_dir": None,
    # DA4FE's frequency branch requires these fields.  They are overwritten
    # by run_params.json/checkpoint args for newer runs, and provide a safe
    # fallback for older checkpoints whose metadata did not record them.
    "sampling_rate": 1000.0,
    "frequency_window": "hann",
    "frequency_normalization": "relative",
    "seq_len": 512,
    "requested_seq_len": 512,
    "patch_len": 4,
    "enc_in": 128,
    "d_model": 256,
    "n_heads": 4,
    "t_layer": 2,
    "v_layer": 2,
    "f_layer": 2,
    "da4fe_channel_dim": 128,
    "da4fe_temporal_dim": 128,
    "da4fe_frequency_dim": 128,
    "da4fe_fusion_mode": "concat_mlp",
    "da4fe_fusion_hidden_dim": 256,
    "da4fe_fusion_out_dim": 256,
    "da4fe_channel_weight": 1.0,
    "da4fe_temporal_weight": 1.0,
    "da4fe_frequency_weight": 1.0,
    "dropout": 0.3,
    "augmentations": "flip0.8,frequency0.,jitter0.,mask0.0,channel0.4,drop0.0",
    "eeg_normalize": False,
    "eeg_num_classes": 0,
    "eeg_adaptive_seq_len": True,
    "batch_size": 16,
    "num_workers": 0,
    "seed": 42,
    "stage1_loss": "triplet",
    "stage1_triplet_type": "semihard",
    "stage1_triplet_margin": 0.2,
    "stage1_ce_weight": 1.0,
    "stage1_triplet_weight": 1.0,
    "stage1_label_smoothing": 0.0,
    "stage1_arcface_s": 30.0,
    "stage1_arcface_m": 0.5,
    "stage1_cosface_s": 30.0,
    "stage1_cosface_m": 0.35,
    "stage1_kmeans_clusters": 0,
    "stage1_samples_per_class": 2,
    "use_validation": True,
    "result_dir": None,
    # Exp_Basic expects these attributes even though inference does not use
    # the optimizer/training settings.
    "use_gpu": False,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0",
    "device_ids": [0],
    "use_amp": False,
}


def parse_bool(value: Any, default: bool = True) -> bool:
    """Parse JSON/CLI-style booleans without treating ``"false"`` as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Stage-1 checkpoint, extract train/val/test embeddings, "
            "and draw a 2-D feature distribution."
        )
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "--weight",
        "--model-weights",
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "checkpoint.pth (or model weights), or a Stage-1 run directory "
            "containing checkpoint.pth"
        ),
    )
    parser.add_argument(
        "--run-params",
        "--run_params",
        dest="run_params",
        type=Path,
        default=None,
        help=(
            "training run_params.json; defaults to the checkpoint directory "
            "and is read automatically"
        ),
    )
    parser.add_argument(
        "--split-summary",
        "--split_summary",
        dest="split_summary",
        type=Path,
        default=None,
        help=(
            "training split_summary.json; defaults to the checkpoint directory "
            "and is read automatically"
        ),
    )
    # Backward-compatible alias for the previous script interface.  It is
    # treated as an explicit run_params.json path.
    parser.add_argument(
        "--args-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("feature_plots"))

    # Common configuration overrides.  All other model/data arguments are
    # recovered from checkpoint['args'] (or run_params.json).
    parser.add_argument("--data", default=None)
    parser.add_argument("--root-path", "--root_path", dest="root_path", default=None)
    parser.add_argument("--data-path", "--data_path", dest="data_path", default=None)
    parser.add_argument("--eeg-hf-dataset-id", dest="eeg_hf_dataset_id", default=None)
    parser.add_argument("--eeg-hf-cache-dir", dest="eeg_hf_cache_dir", default=None)
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=None)
    parser.add_argument(
        "--sampling-rate",
        "--sampling_rate",
        dest="sampling_rate",
        type=float,
        default=None,
        help="EEG sampling rate in Hz; overrides metadata when supplied",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="cpu is the safe default; use cuda explicitly when GPU inference is desired",
    )

    parser.add_argument("--method", choices=("tsne", "pca"), default="tsne")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="maximum points plotted per split; 0 plots every extracted sample",
    )
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-class-names",
        action="store_true",
        help="use dataset class names in the legend instead of numeric labels",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="do not L2-normalize embeddings before saving/reduction",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def read_json_args(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if isinstance(payload.get("args"), dict):
        values = dict(payload["args"])
        # Newer runs record this at the metadata top level as well as in
        # args.  Preserve it for checkpoints that omitted the args field.
        if "validation_enabled" in payload:
            values["use_validation"] = payload["validation_enabled"]
        return values
    return dict(payload)


def resolve_metadata_path(
    checkpoint_path: Path, explicit_path: Path | None, filename: str
) -> Path | None:
    if explicit_path is not None:
        explicit_path = explicit_path.expanduser()
        if not explicit_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {explicit_path}")
        return explicit_path
    candidate = checkpoint_path.parent / filename
    return candidate if candidate.exists() else None


def resolve_split_summary_from_run_params(run_params_path: Path | None) -> Path | None:
    """Find split_summary.json when training used a separate log directory."""
    if run_params_path is None or not run_params_path.exists():
        return None
    try:
        payload = json.loads(run_params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    paths = payload.get("paths") if isinstance(payload, dict) else None
    log_dir = paths.get("log_dir") if isinstance(paths, dict) else None
    if not log_dir:
        return None
    candidate = Path(log_dir).expanduser() / "split_summary.json"
    return candidate if candidate.exists() else None


def load_checkpoint_and_args(
    checkpoint_path: Path,
    run_params_path: Path | None,
    split_summary_path: Path | None,
    legacy_args_path: Path | None,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        checkpoint_args = dict(checkpoint["args"])

    # The training code writes both files next to checkpoint.pth.  Prefer the
    # explicit metadata files so this extractor remains usable even when a
    # checkpoint was saved without embedded args.
    metadata_args: dict[str, Any] = {}
    if legacy_args_path is not None and run_params_path is None:
        run_params_path = legacy_args_path
    for metadata_path in (split_summary_path, run_params_path):
        if metadata_path is not None:
            metadata_args.update(read_json_args(metadata_path))

    saved_args = dict(checkpoint_args)
    saved_args.update(metadata_args)
    return checkpoint, saved_args


def build_runtime_args(
    saved_args: dict[str, Any], cli: argparse.Namespace
) -> SimpleNamespace:
    values = dict(DEFAULT_ARGS)
    values.update(saved_args)

    overrides = {
        "data": cli.data,
        "root_path": cli.root_path,
        "data_path": cli.data_path,
        "eeg_hf_dataset_id": cli.eeg_hf_dataset_id,
        "eeg_hf_cache_dir": cli.eeg_hf_cache_dir,
        "seq_len": cli.seq_len,
        "sampling_rate": cli.sampling_rate,
        "batch_size": cli.batch_size,
        "num_workers": cli.num_workers,
        "seed": cli.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    if values.get("requested_seq_len") in (None, 0):
        values["requested_seq_len"] = values.get("seq_len", 512)

    # Older run metadata may contain ``sampling_rate: null`` (or omit it),
    # while DA4FE's frequency branch rejects a non-positive value during model
    # construction.  The training entry point uses 1000 Hz by default, so
    # restore that value only when it is needed and invalid/missing.
    if str(values.get("model", "")).upper() == "DA4FE":
        try:
            sampling_rate = float(values.get("sampling_rate"))
        except (TypeError, ValueError):
            sampling_rate = 0.0
        raw_f_layer = values.get("f_layer")
        if raw_f_layer is None:
            # DA4FE falls back to t_layer when f_layer is omitted/None.
            raw_f_layer = values.get("t_layer", 0)
        try:
            frequency_branch_enabled = int(raw_f_layer or 0) > 0
        except (TypeError, ValueError):
            frequency_branch_enabled = False
        if frequency_branch_enabled and (
            not np.isfinite(sampling_rate) or sampling_rate <= 0
        ):
            values["sampling_rate"] = 1000.0
            print(
                "Warning: sampling_rate is missing or non-positive in the saved "
                "metadata; using the training default 1000 Hz."
            )
        else:
            values["sampling_rate"] = sampling_rate

        if frequency_branch_enabled:
            frequency_window = str(values.get("frequency_window", "")).lower()
            if frequency_window not in {"hann", "rectangular"}:
                values["frequency_window"] = "hann"
                print(
                    "Warning: frequency_window is missing or invalid in the saved "
                    "metadata; using 'hann'."
                )
            else:
                values["frequency_window"] = frequency_window
            frequency_normalization = str(
                values.get("frequency_normalization", "")
            ).lower()
            if frequency_normalization not in {"relative", "physical"}:
                values["frequency_normalization"] = "relative"
                print(
                    "Warning: frequency_normalization is missing or invalid in the "
                    "saved metadata; using 'relative'."
                )
            else:
                values["frequency_normalization"] = frequency_normalization

    values["use_validation"] = parse_bool(
        values.get("use_validation", True), default=True
    )
    values["num_workers"] = max(0, int(values.get("num_workers", 0)))
    values["seed"] = int(values.get("seed", 42))

    # The extractor uses a single device.  DataParallel prefixes are handled
    # by load_model_state, so there is no need to initialize multiple GPUs.
    requested_gpu = bool(values.get("use_gpu", False))
    if cli.device == "cpu":
        # Some containerized/virtual GPU runtimes crash even when only
        # torch.cuda.is_available() is queried.  Hide CUDA before any such
        # query so an explicit CPU request is genuinely CPU-only.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        use_gpu = False
    elif cli.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        use_gpu = True
    else:
        use_gpu = requested_gpu and torch.cuda.is_available()
    values["use_gpu"] = use_gpu
    values["use_multi_gpu"] = False
    values["device_ids"] = [int(values.get("gpu", 0))]
    values["devices"] = str(values.get("gpu", 0))
    values["gpu"] = int(values.get("gpu", 0))

    # Avoid accidental stochasticity in the dataset split and model layers.
    seed = values["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return SimpleNamespace(**values)


def extract_split(
    exp: Exp_Stage1_Feature,
    split: str,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    dataset, _ = exp._get_data(flag=split)
    return extract_dataset(exp, dataset, normalize=normalize)


def extract_dataset(
    exp: Exp_Stage1_Feature,
    dataset,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Extract embeddings from an already constructed dataset object."""
    loader = exp._build_eval_loader(dataset)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    exp.model.eval()
    with torch.inference_mode():
        for batch_x, batch_label, _padding_mask in loader:
            batch_x = batch_x.float().to(exp.device)
            _logits, fused_features = exp.model(batch_x, return_features=True)
            embedding = fused_features
            if normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
            features.append(embedding.cpu().numpy().astype(np.float32, copy=False))
            labels.append(batch_label.detach().cpu().numpy().reshape(-1).astype(np.int64))

    if not features:
        raise RuntimeError("No samples found in the requested dataset")
    class_names = getattr(dataset, "class_names", None)
    if class_names is not None:
        class_names = [str(name) for name in class_names]
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), class_names


def choose_plot_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(labels) <= max_points:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    # Stratification keeps small classes visible in the plot.
    chosen: list[int] = []
    classes = np.unique(labels)
    per_class = max(1, max_points // max(1, len(classes)))
    for label in classes:
        class_indices = np.flatnonzero(labels == label)
        count = min(per_class, len(class_indices))
        chosen.extend(rng.choice(class_indices, size=count, replace=False).tolist())
    if len(chosen) < max_points:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen, dtype=int))
        count = min(max_points - len(chosen), len(remaining))
        if count:
            chosen.extend(rng.choice(remaining, size=count, replace=False).tolist())
    return np.asarray(sorted(chosen[:max_points]), dtype=int)


def reduce_features(
    all_features: np.ndarray,
    method: str,
    pca_dim: int,
    perplexity: float,
    seed: int,
) -> np.ndarray:
    if all_features.ndim != 2 or len(all_features) < 3:
        raise ValueError("At least three 2-D feature vectors are required for plotting")

    reduced_input = all_features
    if method == "tsne" and pca_dim > 0 and all_features.shape[1] > pca_dim:
        n_components = min(int(pca_dim), all_features.shape[1], len(all_features) - 1)
        reduced_input = PCA(n_components=n_components, random_state=seed).fit_transform(
            all_features
        )

    if method == "pca":
        n_components = min(2, reduced_input.shape[1], len(reduced_input) - 1)
        coordinates = PCA(n_components=n_components, random_state=seed).fit_transform(
            reduced_input
        )
    else:
        safe_perplexity = min(float(perplexity), float(len(reduced_input) - 1))
        safe_perplexity = max(2.0, safe_perplexity)
        coordinates = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(reduced_input)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
    return coordinates.astype(np.float32, copy=False)


def make_palette(num_classes: int) -> list[tuple[float, float, float]]:
    if num_classes <= 0:
        return []
    return [tuple(color) for color in sns.color_palette("husl", n_colors=num_classes)]


def plot_distribution(
    coordinates: np.ndarray,
    split_slices: dict[str, slice],
    labels_by_split: dict[str, np.ndarray],
    class_names: list[str] | None,
    output_path: Path,
    method: str,
    point_size: float,
    alpha: float,
    dpi: int,
    combined: bool,
) -> None:
    all_labels = np.concatenate(list(labels_by_split.values()))
    num_classes = max(
        len(class_names or []), int(all_labels.max()) + 1 if len(all_labels) else 0
    )
    palette = make_palette(num_classes)
    sns.set_theme(style="darkgrid", context="notebook")

    if combined:
        panel_count = max(1, len(split_slices))
        fig, axes = plt.subplots(
            1, panel_count, figsize=(5.0 * panel_count, 4.5), squeeze=False
        )
        axes_list = list(np.atleast_1d(axes[0]))
        fig.subplots_adjust(left=0.04, right=0.80, bottom=0.13, top=0.88, wspace=0.22)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        axes_list = [ax]
        fig.subplots_adjust(left=0.10, right=0.78, bottom=0.11, top=0.90)

    split_order = list(split_slices)
    for axis, split in zip(axes_list, split_order):
        current_slice = split_slices[split]
        split_coordinates = coordinates[current_slice]
        split_labels = labels_by_split[split]
        for label in range(num_classes):
            mask = split_labels == label
            if not np.any(mask):
                continue
            axis.scatter(
                split_coordinates[mask, 0],
                split_coordinates[mask, 1],
                s=point_size,
                alpha=alpha,
                color=palette[label],
                edgecolors="none",
                rasterized=True,
            )
        axis.set_title(split.replace("val", "validation").title())
        axis.set_xlabel(f"{method.upper()}-1")
        axis.set_ylabel(f"{method.upper()}-2")
        axis.grid(True, linewidth=0.6, alpha=0.65)

    legend_labels = class_names if class_names and len(class_names) >= num_classes else [
        str(index) for index in range(num_classes)
    ]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[index],
            markeredgecolor="none",
            markersize=5,
            label=legend_labels[index],
        )
        for index in range(num_classes)
    ]
    if handles:
        fig.legend(
            handles=handles,
            labels=legend_labels[:num_classes],
            loc="center left",
            bbox_to_anchor=(0.81, 0.50),
            ncol=2 if num_classes > 20 else 1,
            fontsize=7.5,
            frameon=False,
            handletextpad=0.3,
            columnspacing=0.8,
        )
    fig.suptitle("Stage-1 feature distribution", y=0.98)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    checkpoint_path = resolve_checkpoint(cli.checkpoint)
    run_params_path = resolve_metadata_path(
        checkpoint_path, cli.run_params, "run_params.json"
    )
    split_summary_path = resolve_metadata_path(
        checkpoint_path, cli.split_summary, "split_summary.json"
    )
    if split_summary_path is None:
        split_summary_path = resolve_split_summary_from_run_params(run_params_path)
    checkpoint, saved_args = load_checkpoint_and_args(
        checkpoint_path,
        run_params_path,
        split_summary_path,
        cli.args_json,
    )
    runtime_args = build_runtime_args(saved_args, cli)

    print(f"Loading checkpoint: {checkpoint_path}")
    if run_params_path is not None:
        print(f"Loaded run parameters: {run_params_path}")
    if split_summary_path is not None:
        print(f"Loaded split summary: {split_summary_path}")
    print(f"Using device: {'cuda' if runtime_args.use_gpu else 'cpu'}")
    exp = Exp_Stage1_Feature(runtime_args)
    load_model_state(exp.model, checkpoint, map_location=exp.device)
    exp.model.eval()

    split_data: dict[str, tuple[np.ndarray, np.ndarray, list[str] | None]] = {}
    validation_enabled = parse_bool(
        getattr(runtime_args, "use_validation", True), default=True
    )
    if validation_enabled:
        split_names = ("TRAIN", "VAL", "TEST")
        for split in split_names:
            features, labels, class_names = extract_split(
                exp, split, normalize=not cli.no_normalize
            )
            split_data[split.lower()] = (features, labels, class_names)
    else:
        # Stage-1 no-validation training folds the original validation samples
        # into training.  Reproduce that exact split here so the plotted
        # training distribution corresponds to what the checkpoint saw.
        raw_train_data, _ = exp._get_data(flag="TRAIN")
        raw_val_data, _ = exp._get_data(flag="VAL")
        merged_train_data = exp._merge_datasets(raw_train_data, raw_val_data)
        features, labels, class_names = extract_dataset(
            exp, merged_train_data, normalize=not cli.no_normalize
        )
        split_data["train"] = (features, labels, class_names)
        test_features, test_labels, test_class_names = extract_split(
            exp, "TEST", normalize=not cli.no_normalize
        )
        split_data["test"] = (test_features, test_labels, test_class_names)

    print(f"Validation enabled: {validation_enabled}")
    for split, (features, labels, _class_names) in split_data.items():
        print(
            f"{split.upper():<5}: samples={len(features)}, feature_dim={features.shape[1]}, "
            f"classes={len(np.unique(labels))}"
        )

    output_dir = cli.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not validation_enabled:
        # Avoid leaving a stale validation-only image from an earlier run in
        # the same output directory, which could be mistaken for this result.
        stale_val_plot = output_dir / "feature_distribution_val.png"
        if stale_val_plot.exists():
            stale_val_plot.unlink()
            print(f"Removed stale validation plot: {stale_val_plot}")

    class_names = next(
        (entry[2] for entry in split_data.values() if entry[2] is not None), None
    )
    features_npz = output_dir / "features.npz"
    np.savez_compressed(
        features_npz,
        **{
            f"{split}_features": values[0]
            for split, values in split_data.items()
        },
        **{
            f"{split}_labels": values[1]
            for split, values in split_data.items()
        },
    )

    # Subsample only the plotted points; features.npz always contains every
    # sample from all available splits.
    plotted_features: list[np.ndarray] = []
    plotted_labels: dict[str, np.ndarray] = {}
    split_slices: dict[str, slice] = {}
    offset = 0
    for index, (split, (features, labels, _names)) in enumerate(split_data.items()):
        selected = choose_plot_indices(labels, cli.max_points, runtime_args.seed + index)
        selected_features = features[selected]
        plotted_features.append(selected_features)
        plotted_labels[split] = labels[selected]
        split_slices[split] = slice(offset, offset + len(selected_features))
        offset += len(selected_features)

    all_plotted = np.concatenate(plotted_features, axis=0)
    coordinates = reduce_features(
        all_plotted,
        method=cli.method,
        pca_dim=cli.pca_dim,
        perplexity=cli.perplexity,
        seed=runtime_args.seed,
    )
    np.savez_compressed(
        output_dir / "reduced_coordinates.npz",
        coordinates=coordinates,
        **{f"{split}_labels": labels for split, labels in plotted_labels.items()},
    )

    plot_distribution(
        coordinates,
        split_slices,
        plotted_labels,
        class_names if cli.show_class_names else None,
        output_dir / "feature_distribution_all.png",
        method=cli.method,
        point_size=cli.point_size,
        alpha=cli.alpha,
        dpi=cli.dpi,
        combined=True,
    )
    for split in split_slices:
        local_coordinates = coordinates[split_slices[split]]
        plot_distribution(
            local_coordinates,
            {split: slice(0, len(local_coordinates))},
            {split: plotted_labels[split]},
            class_names if cli.show_class_names else None,
            output_dir / f"feature_distribution_{split}.png",
            method=cli.method,
            point_size=cli.point_size,
            alpha=cli.alpha,
            dpi=cli.dpi,
            combined=False,
        )

    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "run_params": None if run_params_path is None else str(run_params_path.resolve()),
        "split_summary": None
        if split_summary_path is None
        else str(split_summary_path.resolve()),
        "method": cli.method,
        "normalized": not cli.no_normalize,
        "validation_enabled": validation_enabled,
        "plotted_splits": list(split_data),
        "max_points_per_split": cli.max_points,
        "seed": runtime_args.seed,
        "splits": {
            split: {
                "samples": int(len(values[0])),
                "feature_dim": int(values[0].shape[1]),
                "classes": int(len(np.unique(values[1]))),
                "plotted_samples": int(len(plotted_labels[split])),
            }
            for split, values in split_data.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved embeddings: {features_npz}")
    print(f"Saved plots and coordinates under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
