"""Cluster Stage-1 EEG embeddings and summarize wrongly clustered samples.

The script loads the same model/checkpoint configuration as
``plot_stage1_features.py``.  It runs KMeans over the extracted embeddings,
assigns the most frequent true label in each cluster as that cluster's main
class, and treats all other samples in the cluster as wrongly clustered.

Example::

    python scripts/analyze_stage1_clustering.py \
        --checkpoint ../result/DA4FE/stage1/<run>/checkpoint.pth \
        --output-dir clustering_analysis \
        --device cpu

Outputs include cluster summaries, a row for every wrongly clustered sample,
and subject/class error statistics (CSV and JSON).  If the run was trained
with ``use_validation=false``, the original validation samples are merged into
the train split and only train/test are analyzed, matching training and the
feature-plotting script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.exp_stage1_feature import Exp_Stage1_Feature  # noqa: E402
from utils.checkpointing import load_model_state  # noqa: E402
from scripts.plot_stage1_features import (  # noqa: E402
    build_runtime_args,
    load_checkpoint_and_args,
    parse_bool,
    resolve_checkpoint,
    resolve_metadata_path,
    resolve_split_summary_from_run_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract EEG embeddings, run KMeans clustering, and summarize "
            "wrong cluster assignments by subject and true class."
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
        help="checkpoint.pth or a Stage-1 run directory containing it",
    )
    parser.add_argument("--run-params", "--run_params", dest="run_params", type=Path)
    parser.add_argument(
        "--split-summary", "--split_summary", dest="split_summary", type=Path
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("clustering_analysis"),
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=40,
        help="number of KMeans clusters (default: 40)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=None
    )
    parser.add_argument(
        "--sampling-rate",
        "--sampling_rate",
        dest="sampling_rate",
        type=float,
        default=None,
        help="EEG sampling rate in Hz; overrides metadata when supplied",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="cpu is the safe default; use cuda explicitly when available",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="do not L2-normalize fused features before clustering",
    )
    parser.add_argument("--stage1_l2_normalize", "--stage1-l2-normalize", type=parse_bool, default=None, help="override final Stage-1 feature L2 normalization")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help=(
            "maximum samples used by KMeans; 0 uses all samples. If set, "
            "sampling is stratified by split and true class."
        ),
    )
    return parser.parse_args()


def _scalar_to_text(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return str(value[0])
        return ",".join(str(item) for item in value)
    return str(value)


def dataset_subjects(dataset) -> list[str]:
    """Return subject IDs in exactly the dataset indexing order."""
    # MergedStage1Dataset/ConcatDataset: recurse through components.
    if hasattr(dataset, "datasets"):
        subjects: list[str] = []
        for component in dataset.datasets:
            subjects.extend(dataset_subjects(component))
        if len(subjects) != len(dataset):
            raise RuntimeError("Merged dataset subject count does not match its length")
        return subjects

    # Local EEG-ImageNet loader stores (path, label, subject) tuples.
    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        subjects = [_scalar_to_text(sample[2]) for sample in samples]
        if len(subjects) == len(dataset):
            return subjects

    # Hugging Face EEG-ImageNet loader keeps `subject` in the underlying
    # datasets.Dataset.  Indexing it has the same order as __getitem__.
    underlying = getattr(dataset, "dataset", None)
    if underlying is not None and hasattr(underlying, "__len__"):
        subjects = []
        for index in range(len(underlying)):
            row = underlying[index]
            if isinstance(row, dict) and "subject" in row:
                subjects.append(_scalar_to_text(row["subject"]))
            else:
                subjects = []
                break
        if len(subjects) == len(dataset):
            return subjects

    for attribute in ("subjects", "subject", "subject_ids"):
        values = getattr(dataset, attribute, None)
        if values is not None:
            subjects = [_scalar_to_text(value) for value in values]
            if len(subjects) == len(dataset):
                return subjects

    raise RuntimeError(
        "Cannot recover subject IDs from this dataset. Expected local EEG-ImageNet "
        "samples or a Hugging Face dataset with a `subject` column."
    )


def extract_dataset(exp: Exp_Stage1_Feature, dataset, normalize: bool):
    """Extract features, labels, and subjects from one dataset."""
    subjects = dataset_subjects(dataset)
    loader = exp._build_eval_loader(dataset)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []

    exp.model.eval()
    with torch.inference_mode():
        for batch_x, batch_label, _padding_mask in loader:
            batch_x = batch_x.float().to(exp.device)
            _logits, fused_features = exp.model(batch_x, return_features=True)
            if normalize:
                fused_features = F.normalize(fused_features, p=2, dim=1)
            feature_rows.append(fused_features.cpu().numpy().astype(np.float32, copy=False))
            label_rows.append(batch_label.detach().cpu().numpy().reshape(-1).astype(np.int64))

    if not feature_rows:
        raise RuntimeError("No samples found in the requested dataset")
    features = np.concatenate(feature_rows, axis=0)
    labels = np.concatenate(label_rows, axis=0)
    if len(features) != len(subjects) or len(labels) != len(subjects):
        raise RuntimeError(
            f"Feature/label/subject count mismatch: {len(features)}/{len(labels)}/{len(subjects)}"
        )
    class_names = getattr(dataset, "class_names", None)
    class_names = None if class_names is None else [str(name) for name in class_names]
    return features, labels, np.asarray(subjects, dtype=object), class_names


def stratified_indices(
    split_labels: dict[str, np.ndarray], max_points: int, seed: int
) -> dict[str, np.ndarray]:
    if max_points <= 0:
        return {split: np.arange(len(labels)) for split, labels in split_labels.items()}
    total = sum(len(labels) for labels in split_labels.values())
    if total <= max_points:
        return {split: np.arange(len(labels)) for split, labels in split_labels.items()}

    rng = np.random.default_rng(seed)
    groups: list[tuple[str, int, np.ndarray]] = []
    for split, labels in split_labels.items():
        for label in np.unique(labels):
            groups.append((split, int(label), np.flatnonzero(labels == label)))

    # Give every nonempty split/class group a chance, then fill remaining
    # slots uniformly from the not-yet-selected samples.
    selected: dict[str, list[int]] = {split: [] for split in split_labels}
    for split, _label, indices in groups:
        selected[split].append(int(rng.choice(indices)))
    selected_count = sum(len(values) for values in selected.values())
    if selected_count > max_points:
        flat_groups = [(split, index) for split, values in selected.items() for index in values]
        keep = rng.choice(len(flat_groups), size=max_points, replace=False)
        selected = {split: [] for split in split_labels}
        for flat_index in keep:
            split, index = flat_groups[int(flat_index)]
            selected[split].append(index)
    else:
        remaining = [
            (split, index)
            for split, labels in split_labels.items()
            for index in range(len(labels))
            if index not in set(selected[split])
        ]
        extra_count = min(max_points - selected_count, len(remaining))
        if extra_count:
            for flat_index in rng.choice(len(remaining), size=extra_count, replace=False):
                split, index = remaining[int(flat_index)]
                selected[split].append(index)
    return {split: np.asarray(sorted(values), dtype=int) for split, values in selected.items()}


def percentage(series: pd.Series, denominator: int) -> pd.Series:
    if denominator <= 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return series.astype(float) * 100.0 / float(denominator)


def main() -> None:
    cli = parse_args()
    if cli.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2")
    if cli.max_points < 0:
        raise ValueError("--max-points must be >= 0 (0 means use all samples)")
    if cli.batch_size is not None and cli.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if cli.num_workers is not None and cli.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    checkpoint_path = resolve_checkpoint(cli.checkpoint)
    run_params_path = resolve_metadata_path(checkpoint_path, cli.run_params, "run_params.json")
    split_summary_path = resolve_metadata_path(
        checkpoint_path, cli.split_summary, "split_summary.json"
    )
    if split_summary_path is None:
        split_summary_path = resolve_split_summary_from_run_params(run_params_path)
    checkpoint, saved_args = load_checkpoint_and_args(
        checkpoint_path, run_params_path, split_summary_path, None
    )

    cli_for_runtime = SimpleNamespace(
        data=None,
        root_path=None,
        data_path=None,
        eeg_hf_dataset_id=None,
        eeg_hf_cache_dir=None,
        seq_len=None,
        sampling_rate=cli.sampling_rate,
        stage1_l2_normalize=cli.stage1_l2_normalize,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        seed=cli.seed,
        device=cli.device,
    )
    runtime_args = build_runtime_args(saved_args, cli_for_runtime)
    if cli.no_normalize:
        # Keep the legacy --no-normalize switch authoritative for both the
        # model's returned feature and the KMeans input.
        runtime_args.stage1_l2_normalize = False
    feature_normalize = bool(runtime_args.stage1_l2_normalize)

    print(f"Loading checkpoint: {checkpoint_path}")
    if run_params_path is not None:
        print(f"Loaded run parameters: {run_params_path}")
    if split_summary_path is not None:
        print(f"Loaded split summary: {split_summary_path}")
    print(f"Using device: {'cuda' if runtime_args.use_gpu else 'cpu'}")
    print(f"Final feature L2 normalize: {feature_normalize}")

    exp = Exp_Stage1_Feature(runtime_args)
    load_model_state(exp.model, checkpoint, map_location=exp.device)
    exp.model.eval()

    validation_enabled = parse_bool(
        getattr(runtime_args, "use_validation", True), default=True
    )
    split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]] = {}
    if validation_enabled:
        for split in ("TRAIN", "VAL", "TEST"):
            dataset, _ = exp._get_data(flag=split)
            split_data[split.lower()] = extract_dataset(
                exp, dataset, normalize=feature_normalize
            )
    else:
        raw_train, _ = exp._get_data(flag="TRAIN")
        raw_val, _ = exp._get_data(flag="VAL")
        merged_train = exp._merge_datasets(raw_train, raw_val)
        split_data["train"] = extract_dataset(
            exp, merged_train, normalize=feature_normalize
        )
        test_data, _ = exp._get_data(flag="TEST")
        split_data["test"] = extract_dataset(
            exp, test_data, normalize=feature_normalize
        )

    for split, (features, labels, subjects, _class_names) in split_data.items():
        print(
            f"{split.upper():<5}: samples={len(features)}, feature_dim={features.shape[1]}, "
            f"classes={len(np.unique(labels))}, subjects={len(np.unique(subjects))}"
        )

    class_names = next(
        (values[3] for values in split_data.values() if values[3] is not None), None
    )
    num_classes = max(
        len(class_names or []),
        max(int(values[1].max()) for values in split_data.values()) + 1,
    )
    if num_classes != cli.n_clusters:
        print(
            f"Warning: clustering {num_classes} true classes into {cli.n_clusters} clusters."
        )

    split_labels = {split: values[1] for split, values in split_data.items()}
    selected_by_split = stratified_indices(split_labels, cli.max_points, runtime_args.seed)
    selected_features = []
    rows: list[dict[str, Any]] = []
    for split, (features, labels, subjects, _names) in split_data.items():
        indices = selected_by_split[split]
        selected_features.append(features[indices])
        for index in indices:
            rows.append(
                {
                    "split": split,
                    "sample_index": int(index),
                    "subject": str(subjects[index]),
                    "true_label": int(labels[index]),
                }
            )

    all_features = np.concatenate(selected_features, axis=0)
    all_rows = pd.DataFrame(rows)
    if len(all_features) < cli.n_clusters:
        raise ValueError(
            f"Need at least {cli.n_clusters} samples for KMeans, got {len(all_features)}"
        )
    if not np.isfinite(all_features).all():
        raise ValueError(
            "Extracted features contain NaN or infinity; check the checkpoint and EEG data "
            "before running KMeans."
        )

    kmeans = KMeans(
        n_clusters=cli.n_clusters,
        random_state=runtime_args.seed,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(all_features)
    all_rows["cluster_id"] = cluster_ids.astype(int)

    dominant_labels: dict[int, int] = {}
    cluster_records: list[dict[str, Any]] = []
    for cluster_id in range(cli.n_clusters):
        mask = cluster_ids == cluster_id
        cluster_labels = all_rows.loc[mask, "true_label"].to_numpy(dtype=int)
        counts = np.bincount(cluster_labels, minlength=num_classes)
        dominant_label = int(np.argmax(counts))
        dominant_count = int(counts[dominant_label])
        dominant_labels[cluster_id] = dominant_label
        cluster_size = int(mask.sum())
        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "sample_count": cluster_size,
                "main_label": dominant_label,
                "main_label_name": (
                    class_names[dominant_label]
                    if class_names is not None and dominant_label < len(class_names)
                    else str(dominant_label)
                ),
                "main_label_count": dominant_count,
                "main_label_ratio_pct": 100.0 * dominant_count / cluster_size
                if cluster_size
                else 0.0,
                # Keep the complete composition available without widening
                # the CSV into 40 separate columns.  It is especially useful
                # when checking why a sample was marked as wrong.
                "label_distribution": json.dumps(
                    {
                        str(label): int(count)
                        for label, count in enumerate(counts)
                        if count > 0
                    },
                    ensure_ascii=False,
                ),
                "wrong_count": cluster_size - dominant_count,
                "wrong_ratio_pct": 100.0 * (cluster_size - dominant_count) / cluster_size
                if cluster_size
                else 0.0,
            }
        )

    all_rows["cluster_main_label"] = all_rows["cluster_id"].map(dominant_labels).astype(int)
    all_rows["cluster_main_label_name"] = all_rows["cluster_main_label"].map(
        lambda label: class_names[label]
        if class_names is not None and label < len(class_names)
        else str(label)
    )
    all_rows["true_label_name"] = all_rows["true_label"].map(
        lambda label: class_names[label]
        if class_names is not None and label < len(class_names)
        else str(label)
    )
    all_rows["is_wrong"] = all_rows["true_label"] != all_rows["cluster_main_label"]
    wrong_rows = all_rows.loc[all_rows["is_wrong"]].copy()

    output_dir = cli.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output_dir / "clustering_results.csv", index=False, encoding="utf-8-sig")
    wrong_rows.to_csv(output_dir / "wrong_cluster_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cluster_records).to_csv(
        output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig"
    )

    total_wrong = len(wrong_rows)
    subject_stats = (
        all_rows.groupby("subject", dropna=False)
        .agg(total_count=("subject", "size"), wrong_count=("is_wrong", "sum"))
        .reset_index()
    )
    subject_stats["error_rate_pct"] = (
        subject_stats["wrong_count"] * 100.0 / subject_stats["total_count"].replace(0, np.nan)
    ).fillna(0.0)
    subject_stats["wrong_share_pct"] = percentage(subject_stats["wrong_count"], total_wrong)
    subject_stats = subject_stats.sort_values(
        ["wrong_count", "subject"], ascending=[False, True]
    )
    subject_stats.to_csv(
        output_dir / "subject_error_statistics.csv", index=False, encoding="utf-8-sig"
    )

    class_stats = (
        all_rows.groupby(["true_label", "true_label_name"], dropna=False)
        .agg(total_count=("true_label", "size"), wrong_count=("is_wrong", "sum"))
        .reset_index()
    )
    class_stats["error_rate_pct"] = (
        class_stats["wrong_count"] * 100.0 / class_stats["total_count"].replace(0, np.nan)
    ).fillna(0.0)
    class_stats["wrong_share_pct"] = percentage(class_stats["wrong_count"], total_wrong)
    class_stats = class_stats.sort_values(
        ["wrong_count", "true_label"], ascending=[False, True]
    )
    class_stats.to_csv(
        output_dir / "class_error_statistics.csv", index=False, encoding="utf-8-sig"
    )

    # Save the exact features used by KMeans and only the incorrectly clustered
    # rows, making later inspection/reduction reproducible.
    wrong_indices = np.flatnonzero(all_rows["is_wrong"].to_numpy())
    np.savez_compressed(
        output_dir / "clustering_features.npz",
        features=all_features,
        cluster_ids=cluster_ids,
        true_labels=all_rows["true_label"].to_numpy(dtype=np.int64),
        wrong_features=all_features[wrong_indices],
        wrong_row_indices=wrong_indices.astype(np.int64),
    )

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "run_params": None if run_params_path is None else str(run_params_path.resolve()),
        "split_summary": None
        if split_summary_path is None
        else str(split_summary_path.resolve()),
        "validation_enabled": validation_enabled,
        "analyzed_splits": list(split_data),
        "n_clusters": cli.n_clusters,
        "true_class_count": num_classes,
        "normalized": feature_normalize,
        "stage1_l2_normalize": feature_normalize,
        "kmeans_samples": int(len(all_features)),
        "wrong_samples": int(total_wrong),
        "wrong_rate_pct": 100.0 * total_wrong / len(all_features),
        "seed": int(runtime_args.seed),
        "cluster_inertia": float(kmeans.inertia_),
        "class_names": class_names,
    }
    (output_dir / "clustering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"KMeans clusters: {cli.n_clusters}")
    print(f"Samples analyzed: {len(all_features)}")
    print(f"Wrongly clustered: {total_wrong} ({summary['wrong_rate_pct']:.2f}%)")
    print(f"Saved clustering analysis under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
