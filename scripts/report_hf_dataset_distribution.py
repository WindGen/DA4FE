"""Generate a distribution report for the Hugging Face EEG dataset.

The script only reads dataset metadata (``label``/``label_folder`` and
``subject``), so it does not decode the EEG arrays.  It can therefore be used
before training to verify class balance and subject-wise split leakage.

Examples
--------
Use the project defaults::

    python scripts/report_hf_dataset_distribution.py --output-dir hf_report

Reuse the parameters saved by a training run::

    python scripts/report_hf_dataset_distribution.py \
        --run-params ../result/DA4FE/stage1/<run>/run_params.json \
        --split-summary ../result/DA4FE/stage1/<run>/split_summary.json \
        --output-dir hf_report

The output directory contains CSV tables, ``summary.json``, a Chinese
Markdown report, and PNG plots.  The raw train/validation/test splits are
always reported.  When the run metadata says ``use_validation=false``, an
additional effective-training summary (train + validation) is included in the
Markdown and JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - exercised only without datasets
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


DEFAULT_DATASET_ID = "luigi-s/EEG_Image_CVPR_ALL_subj"
LOGICAL_SPLITS = ("train", "val", "test")
SPLIT_CANDIDATES = {
    "train": ("train",),
    "val": ("validation", "val", "dev"),
    "test": ("test",),
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


def natural_key(value: Any) -> list[Any]:
    """Return a dependency-free natural-sort key (``subject2`` before ``subject10``)."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def natural_sorted(values: Iterable[Any]) -> list[str]:
    return sorted((str(value) for value in values), key=natural_key)


def _jsonable(value: Any) -> Any:
    """Convert NumPy/scalar values to objects accepted by ``json.dumps``."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _scalar_to_text(value: Any) -> str:
    """Make labels/subjects stable, readable strings for tables and plots."""
    if value is None:
        return "<unknown>"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_to_text(value.item())
        return ",".join(_scalar_to_text(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return ",".join(_scalar_to_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    return text if text else "<unknown>"


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _metadata_args(payload: dict[str, Any]) -> dict[str, Any]:
    values = {}
    if isinstance(payload.get("args"), dict):
        values.update(payload["args"])
    # ``validation_enabled`` is written at the top level by current runs.
    for key in ("validation_enabled", "use_validation"):
        if key in payload:
            values["use_validation"] = payload[key]
    return values


def resolve_metadata_paths(
    run_dir: Path | None,
    run_params: Path | None,
    split_summary: Path | None,
) -> tuple[Path | None, Path | None]:
    """Resolve explicit paths, a run directory, or files in the cwd."""
    run_dir = run_dir.expanduser() if run_dir is not None else None
    if run_dir is not None and not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if run_params is None and run_dir is not None:
        candidate = run_dir / "run_params.json"
        run_params = candidate if candidate.exists() else None
    if split_summary is None and run_dir is not None:
        candidate = run_dir / "split_summary.json"
        split_summary = candidate if candidate.exists() else None
    if run_params is None:
        candidate = Path("run_params.json")
        run_params = candidate if candidate.exists() else None
    if split_summary is None:
        candidate = Path("split_summary.json")
        split_summary = candidate if candidate.exists() else None

    if run_params is not None:
        run_params = run_params.expanduser()
        if not run_params.exists():
            raise FileNotFoundError(f"run_params file not found: {run_params}")
    if split_summary is not None:
        split_summary = split_summary.expanduser()
        if not split_summary.exists():
            raise FileNotFoundError(f"split_summary file not found: {split_summary}")

    # A separate log directory is used by some older runs.
    if split_summary is None and run_params is not None:
        try:
            payload = _read_json(run_params)
            paths = payload.get("paths", {})
            log_dir = paths.get("log_dir") if isinstance(paths, dict) else None
            if log_dir:
                candidate = Path(log_dir).expanduser() / "split_summary.json"
                if candidate.exists():
                    split_summary = candidate
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return run_params, split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 Hugging Face EEG 数据集的类别、被试和 split 分布")

    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", type=Path, default=None, help="训练结果目录，自动读取其中的 run_params.json 和 split_summary.json")
    parser.add_argument("--run-params", "--run_params", dest="run_params", type=Path, default=None, help="训练时保存的 run_params.json")
    parser.add_argument("--split-summary", "--split_summary", dest="split_summary", type=Path, default=None, help="训练时保存的 split_summary.json")
    parser.add_argument("--dataset-id", "--dataset_id", dest="dataset_id", default=None, help="覆盖 Hugging Face 数据集 ID")
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", default=None, help="Hugging Face 数据集缓存目录")

    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=Path("hf_dataset_report"), help="报告和图表输出目录")
    parser.add_argument("--max-classes", "--max_classes", dest="max_classes", type=int, default=None, help="只保留训练集自然排序后的前 N 类；0 表示全部类别")
    parser.add_argument("--top-subjects", "--top_subjects", dest="top_subjects", type=int, default=20, help="被试图中每个 split 显示的最多被试数")
    parser.add_argument("--include-effective-train", "--include_effective_train", dest="include_effective_train", type=parse_bool, default=True, help="use_validation=false 时，在 JSON/Markdown 中添加 train+validation 的有效训练集统计")
    return parser.parse_args()


def _dataset_columns(dataset) -> list[str]:
    columns = getattr(dataset, "column_names", None)
    if columns is not None:
        return list(columns)
    if hasattr(dataset, "features"):
        return list(dataset.features.keys())
    raise RuntimeError("Loaded object does not expose Hugging Face dataset columns")


def _label_feature_names(dataset) -> list[str] | None:
    features = getattr(dataset, "features", None)
    feature = features.get("label") if features is not None else None
    names = getattr(feature, "names", None)
    return [str(name) for name in names] if names else None


def _extract_records(dataset, split_name: str) -> list[dict[str, str]]:
    """Read only metadata columns and return one normalized record per row."""
    columns = _dataset_columns(dataset)
    label_column = "label_folder" if "label_folder" in columns else "label"
    if "subject" not in columns:
        warnings.warn(f"Split {split_name!r} 没有 subject 字段，所有样本将计入 <unknown>。")
    metadata_columns = [column for column in (label_column, "label", "subject") if column in columns]
    if not metadata_columns:
        raise RuntimeError(f"Split {split_name!r} has neither label nor label_folder columns")

    label_names = _label_feature_names(dataset)
    try:
        metadata_dataset = dataset.select_columns(metadata_columns)
    except AttributeError:
        metadata_dataset = dataset

    records: list[dict[str, str]] = []
    for row in metadata_dataset:
        raw_label = row.get(label_column)
        if (raw_label is None or _scalar_to_text(raw_label) == "<unknown>") and "label" in row:
            raw_label = row["label"]
        if label_column == "label" and label_names:
            try:
                raw_label = label_names[int(raw_label)]
            except (TypeError, ValueError, IndexError):
                pass
        records.append(
            {
                "split": split_name,
                "class_name": _scalar_to_text(raw_label),
                "subject": _scalar_to_text(row.get("subject")),
            }
        )
    return records


def load_hf_splits(dataset_id: str, cache_dir: str | None) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    if load_dataset is None:
        raise ImportError("需要安装 datasets：pip install datasets") from _DATASETS_IMPORT_ERROR

    records: dict[str, list[dict[str, str]]] = {}
    resolved_names: dict[str, str] = {}
    for logical_split in LOGICAL_SPLITS:
        loaded = None
        last_error: Exception | None = None
        for candidate in SPLIT_CANDIDATES[logical_split]:
            try:
                loaded = load_dataset(dataset_id, split=candidate, cache_dir=cache_dir)
                resolved_names[logical_split] = candidate
                break
            except Exception as exc:  # datasets raises different errors by version
                last_error = exc
        if loaded is None:
            if logical_split == "train":
                raise RuntimeError(f"无法加载 Hugging Face train split ({dataset_id})") from last_error
            warnings.warn(f"未找到 {logical_split} split，将跳过该 split。尝试名称：{SPLIT_CANDIDATES[logical_split]}")
            continue
        records[logical_split] = _extract_records(loaded, logical_split)
    if not records:
        raise RuntimeError(f"数据集 {dataset_id} 没有可用 split")
    return records, resolved_names


def apply_class_filter(records_by_split: dict[str, list[dict[str, str]]], max_classes: int | None) -> tuple[dict[str, list[dict[str, str]]], list[str], int]:
    train_classes = natural_sorted({record["class_name"] for record in records_by_split.get("train", [])})
    if max_classes is None:
        effective_max = 0
    else:
        if max_classes < 0:
            raise ValueError("--max-classes 必须大于等于 0")
        effective_max = int(max_classes)
    if effective_max > len(train_classes):
        raise ValueError(f"--max-classes={effective_max} 超过训练集类别数 {len(train_classes)}")
    class_names = train_classes[:effective_max] if effective_max else train_classes
    allowed = set(class_names)
    filtered = {
        split: [record for record in records if record["class_name"] in allowed]
        for split, records in records_by_split.items()
    }
    return filtered, class_names, effective_max


def _safe_ratio(maximum: int, minimum: int) -> float | None:
    if minimum <= 0:
        return None
    return float(maximum) / float(minimum)


def build_tables(records_by_split: dict[str, list[dict[str, str]]], class_names: list[str]) -> dict[str, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    frames = {
        split: pd.DataFrame(records_by_split.get(split, []), columns=["split", "class_name", "subject"])
        for split in LOGICAL_SPLITS
    }

    for split in LOGICAL_SPLITS:
        frame = frames[split]
        class_counts = frame["class_name"].value_counts().to_dict() if not frame.empty else {}
        subject_counts = frame["subject"].value_counts().to_dict() if not frame.empty else {}
        classes_present = [name for name in class_names if name in class_counts]
        subjects_present = natural_sorted(subject_counts)
        class_values = [int(class_counts[name]) for name in classes_present]
        subject_values = [int(subject_counts[name]) for name in subjects_present]
        split_rows.append(
            {
                "split": split,
                "samples": int(len(frame)),
                "num_classes": int(len(classes_present)),
                "num_subjects": int(len(subjects_present)),
                "min_class_count": int(min(class_values)) if class_values else 0,
                "max_class_count": int(max(class_values)) if class_values else 0,
                "class_imbalance_ratio": _safe_ratio(max(class_values), min(class_values)) if class_values else None,
                "min_subject_count": int(min(subject_values)) if subject_values else 0,
                "max_subject_count": int(max(subject_values)) if subject_values else 0,
                "subject_imbalance_ratio": _safe_ratio(max(subject_values), min(subject_values)) if subject_values else None,
            }
        )
        if not frame.empty:
            grouped_class = frame.groupby("class_name", sort=False)
            grouped_class_subject = frame.groupby(["class_name", "subject"], sort=False).size()
            for class_name, group in grouped_class:
                per_subject = group.groupby("subject").size()
                count = int(len(group))
                class_rows.append(
                    {
                        "split": split,
                        "class_name": str(class_name),
                        "sample_count": count,
                        "percentage": 100.0 * count / len(frame),
                        "subject_count": int(group["subject"].nunique()),
                        "mean_samples_per_subject": float(per_subject.mean()),
                        "min_samples_per_subject": int(per_subject.min()),
                        "max_samples_per_subject": int(per_subject.max()),
                    }
                )
            for subject, group in frame.groupby("subject", sort=False):
                subject_rows.append(
                    {
                        "split": split,
                        "subject": str(subject),
                        "sample_count": int(len(group)),
                        "percentage": 100.0 * len(group) / len(frame),
                        "num_classes": int(group["class_name"].nunique()),
                    }
                )
            for (class_name, subject), count in grouped_class_subject.items():
                cross_rows.append(
                    {
                        "split": split,
                        "class_name": str(class_name),
                        "subject": str(subject),
                        "sample_count": int(count),
                    }
                )

    all_subjects = natural_sorted({record["subject"] for records in records_by_split.values() for record in records})
    overlap_rows: list[dict[str, Any]] = []
    for subject in all_subjects:
        row: dict[str, Any] = {"subject": subject}
        present = 0
        for split in LOGICAL_SPLITS:
            split_frame = frames[split]
            subject_frame = split_frame[split_frame["subject"] == subject] if not split_frame.empty else split_frame
            row[f"{split}_samples"] = int(len(subject_frame))
            row[f"{split}_classes"] = int(subject_frame["class_name"].nunique()) if not subject_frame.empty else 0
            present += int(len(subject_frame) > 0)
        row["num_splits"] = present
        overlap_rows.append(row)

    return {
        "split_summary": pd.DataFrame(split_rows),
        "class_distribution": pd.DataFrame(class_rows),
        "subject_distribution": pd.DataFrame(subject_rows),
        "class_subject_distribution": pd.DataFrame(cross_rows),
        "subject_split_overlap": pd.DataFrame(overlap_rows),
    }


def _ordered_class_frame(class_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    if class_df.empty:
        return class_df
    frame = class_df.copy()
    order = {name: index for index, name in enumerate(class_names)}
    frame["_order"] = frame["class_name"].map(order).fillna(len(order))
    return frame.sort_values(["_order", "split"]).drop(columns="_order")


def plot_class_distribution(class_df: pd.DataFrame, class_names: list[str], output_path: Path) -> None:
    splits = [split for split in LOGICAL_SPLITS if split in set(class_df.get("split", []))]
    if not splits or not class_names:
        return
    pivot = class_df.pivot(index="class_name", columns="split", values="sample_count").reindex(class_names).fillna(0)
    percentage = class_df.pivot(index="class_name", columns="split", values="percentage").reindex(class_names).fillna(0)
    width = min(24.0, max(10.0, 0.34 * len(class_names) + 3.0))
    x = np.arange(len(class_names))
    fig, axes = plt.subplots(2, 1, figsize=(width, 9.0), sharex=True)
    bar_width = 0.8 / max(1, len(splits))
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    for axis, values, ylabel in ((axes[0], pivot, "Samples"), (axes[1], percentage, "Percentage within split (%)")):
        for index, split in enumerate(splits):
            axis.bar(x + (index - (len(splits) - 1) / 2) * bar_width, values.get(split, pd.Series(0, index=class_names)), width=bar_width, label=split, color=colors[split], alpha=0.9)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=70, ha="right", fontsize=8)
    fig.suptitle("Hugging Face EEG class distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_subject_distribution(subject_df: pd.DataFrame, top_subjects: int, output_path: Path) -> None:
    if subject_df.empty:
        return
    splits = [split for split in LOGICAL_SPLITS if split in set(subject_df["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(max(7.0, 4.3 * len(splits)), 5.0), squeeze=False)
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    for axis, split in zip(axes[0], splits):
        current = subject_df[subject_df["split"] == split].sort_values(["sample_count", "subject"], ascending=[False, True]).head(max(1, top_subjects))
        current = current.iloc[::-1]
        axis.barh(current["subject"], current["sample_count"], color=colors[split], alpha=0.9)
        axis.set_title(split)
        axis.set_xlabel("Samples")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Hugging Face EEG subject distribution (top {max(1, top_subjects)} per split)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_subject_heatmap(cross_df: pd.DataFrame, class_names: list[str], output_path: Path) -> None:
    if cross_df.empty:
        return
    pivot = cross_df.pivot_table(index="class_name", columns="subject", values="sample_count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=[name for name in class_names if name in pivot.index])
    pivot = pivot.reindex(columns=natural_sorted(pivot.columns))
    if pivot.empty:
        return
    fig_width = min(26.0, max(9.0, 0.24 * len(pivot.columns) + 3.0))
    fig_height = min(22.0, max(6.0, 0.25 * len(pivot.index) + 2.5))
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_ylabel("Class")
    axis.set_xlabel("Subject")
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index, fontsize=8)
    if len(pivot.columns) <= 60:
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns, rotation=70, ha="right", fontsize=7)
    else:
        axis.set_xticks([])
        axis.set_xlabel(f"Subjects ({len(pivot.columns)} total; see class_subject_distribution.csv for values)")
    fig.colorbar(image, ax=axis, label="Samples")
    axis.set_title("Class × subject sample counts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_ratio(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    return f"{float(value):.2f}"


def write_report(
    output_path: Path,
    dataset_id: str,
    cache_dir: str | None,
    run_params_path: Path | None,
    split_summary_path: Path | None,
    validation_enabled: bool,
    class_names: list[str],
    effective_max_classes: int,
    resolved_names: dict[str, str],
    tables: dict[str, pd.DataFrame],
    records_by_split: dict[str, list[dict[str, str]]],
    include_effective_train: bool,
) -> None:
    summary = tables["split_summary"]
    lines = [
        "# Hugging Face EEG 数据集分布报告",
        "",
        f"- 数据集：`{dataset_id}`",
        f"- 缓存目录：`{cache_dir or '默认缓存'}`",
        f"- run_params：`{run_params_path or '未提供'}`",
        f"- split_summary：`{split_summary_path or '未提供'}`",
        f"- 训练时启用验证集：`{validation_enabled}`",
        f"- 类别过滤：`{'全部类别' if effective_max_classes == 0 else f'训练集自然排序前 {effective_max_classes} 类'}`",
        f"- Hugging Face split 映射：`{resolved_names}`",
        "",
        "## Split 总览",
        "",
        "| split | 样本数 | 类别数 | 被试数 | 类别最小/最大 | 类别不均衡比 | 被试最小/最大 | 被试不均衡比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['split']} | {row['samples']} | {row['num_classes']} | {row['num_subjects']} | {row['min_class_count']}/{row['max_class_count']} | {_format_ratio(row['class_imbalance_ratio'])} | {row['min_subject_count']}/{row['max_subject_count']} | {_format_ratio(row['subject_imbalance_ratio'])} |")

    if include_effective_train and not validation_enabled and "train" in records_by_split:
        effective_records = records_by_split.get("train", []) + records_by_split.get("val", [])
        effective_frame = pd.DataFrame(effective_records, columns=["split", "class_name", "subject"])
        lines.extend(["", "> 当前 run 使用 `use_validation=false`；训练时实际使用的训练样本为 train + validation。原始 validation 仍单独列出，便于审计。", "", f"有效训练集（train+validation）样本数：**{len(effective_frame)}**。"])

    class_df = tables["class_distribution"]
    lines.extend(["", "## 类别分布摘要", ""])
    for split in LOGICAL_SPLITS:
        current = class_df[class_df["split"] == split].sort_values("sample_count", ascending=False)
        if current.empty:
            lines.append(f"- **{split}**：无数据。")
            continue
        top = current.head(10)
        entries = "；".join(f"{row.class_name} ({int(row.sample_count)})" for row in top.itertuples())
        lines.append(f"- **{split}**：最多的类别为 {entries}。完整统计见 `class_distribution.csv`。")

    lines.extend(["", "### 类别覆盖差异", ""])
    for split in ("val", "test"):
        split_class_set = set(class_df.loc[class_df["split"] == split, "class_name"]) if not class_df.empty else set()
        missing = [name for name in class_names if name not in split_class_set]
        if missing:
            preview = "、".join(missing[:20]) + (" …" if len(missing) > 20 else "")
            lines.append(f"- **{split}** 缺少 {len(missing)} 个训练类别：{preview}。")
        else:
            lines.append(f"- **{split}** 覆盖全部训练类别。")

    subject_df = tables["subject_distribution"]
    lines.extend(["", "## 被试分布摘要", ""])
    for split in LOGICAL_SPLITS:
        current = subject_df[subject_df["split"] == split].sort_values("sample_count", ascending=False)
        if current.empty:
            lines.append(f"- **{split}**：无数据。")
            continue
        top = current.head(10)
        entries = "；".join(f"{row.subject} ({int(row.sample_count)})" for row in top.itertuples())
        lines.append(f"- **{split}**：样本最多的被试为 {entries}。完整统计见 `subject_distribution.csv`。")

    lines.extend(["", "## 不均衡提示", ""])
    warnings_added = False
    for row in summary.to_dict("records"):
        if row["class_imbalance_ratio"] is not None and row["class_imbalance_ratio"] >= 2.0:
            lines.append(f"- **{row['split']}** 类别最大/最小样本数比为 {_format_ratio(row['class_imbalance_ratio'])}，存在明显类别不均衡。")
            warnings_added = True
        if row["subject_imbalance_ratio"] is not None and row["subject_imbalance_ratio"] >= 2.0:
            lines.append(f"- **{row['split']}** 被试最大/最小样本数比为 {_format_ratio(row['subject_imbalance_ratio'])}，训练时可考虑按 subject 采样或加权。")
            warnings_added = True
    if not warnings_added:
        lines.append("- 按最大/最小计数比（阈值 2.0）未发现明显不均衡；仍建议结合完整 CSV 检查长尾类别。")

    overlap_df = tables["subject_split_overlap"]
    shared = overlap_df[overlap_df["num_splits"] > 1] if not overlap_df.empty else overlap_df
    pairwise_overlap = {
        f"{left}-{right}": int(((overlap_df[f"{left}_samples"] > 0) & (overlap_df[f"{right}_samples"] > 0)).sum())
        if not overlap_df.empty else 0
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    pairwise_text = "；".join(f"{pair}={count}" for pair, count in pairwise_overlap.items())
    lines.extend(["", "## Split 被试重叠检查", "", f"- 共有被试数：**{len(overlap_df)}**。", f"- 出现在多个 split 的被试数：**{len(shared)}**。", f"- 两两重叠 subject 数：{pairwise_text}。", "- 详细的每被试样本数/类别数见 `subject_split_overlap.csv`。"])
    if len(shared):
        lines.append("- 警告：存在跨 split 的 subject，可能造成被试泄漏；请核对数据集 split 定义。")
    else:
        lines.append("- 未发现跨 split 的 subject（按 `subject` 字段精确匹配）。")

    lines.extend(["", "## 输出文件", "", "- `summary.json`：机器可读的汇总和重叠统计。", "- `split_summary.csv`：每个 split 的样本/类别/被试及不均衡指标。", "- `class_distribution.csv`、`subject_distribution.csv`、`class_subject_distribution.csv`：明细表。", "- `subject_split_overlap.csv`：subject 泄漏检查表。", "- `class_distribution.png`、`subject_distribution.png`、`class_subject_heatmap.png`：可视化。"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    run_params_path, split_summary_path = resolve_metadata_paths(cli.run_dir, cli.run_params, cli.split_summary)
    run_payload = _read_json(run_params_path)
    split_payload = _read_json(split_summary_path)
    metadata = {}
    metadata.update(_metadata_args(run_payload))
    metadata.update(_metadata_args(split_payload))

    dataset_id = str(cli.dataset_id or metadata.get("eeg_hf_dataset_id") or DEFAULT_DATASET_ID)
    cache_dir = cli.cache_dir if cli.cache_dir is not None else metadata.get("eeg_hf_cache_dir")
    if isinstance(cache_dir, str) and cache_dir.strip().lower() in {"", "none", "null"}:
        cache_dir = None
    max_classes = cli.max_classes if cli.max_classes is not None else int(metadata.get("eeg_num_classes") or 0)
    validation_enabled = parse_bool(metadata.get("use_validation", True), default=True)
    if cli.top_subjects <= 0:
        raise ValueError("--top-subjects 必须大于 0")

    print(f"Loading Hugging Face dataset: {dataset_id}")
    records_by_split, resolved_names = load_hf_splits(dataset_id, cache_dir)
    records_by_split, class_names, effective_max_classes = apply_class_filter(records_by_split, max_classes)
    tables = build_tables(records_by_split, class_names)
    tables["class_distribution"] = _ordered_class_frame(tables["class_distribution"], class_names).reset_index(drop=True)
    if not tables["class_subject_distribution"].empty:
        class_order = {name: index for index, name in enumerate(class_names)}
        cross = tables["class_subject_distribution"].copy()
        cross["_class_order"] = cross["class_name"].map(class_order).fillna(len(class_order))
        cross["_subject_order"] = cross["subject"].map({name: index for index, name in enumerate(natural_sorted(cross["subject"]))})
        tables["class_subject_distribution"] = cross.sort_values(["split", "_class_order", "_subject_order"]).drop(columns=["_class_order", "_subject_order"]).reset_index(drop=True)

    output_dir = cli.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    plot_class_distribution(tables["class_distribution"], class_names, output_dir / "class_distribution.png")
    plot_subject_distribution(tables["subject_distribution"], cli.top_subjects, output_dir / "subject_distribution.png")
    plot_class_subject_heatmap(tables["class_subject_distribution"], class_names, output_dir / "class_subject_heatmap.png")

    overlap_df = tables["subject_split_overlap"]
    summary_payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "cache_dir": cache_dir,
        "run_params": None if run_params_path is None else str(run_params_path.resolve()),
        "split_summary": None if split_summary_path is None else str(split_summary_path.resolve()),
        "validation_enabled": validation_enabled,
        "resolved_split_names": resolved_names,
        "max_classes": effective_max_classes,
        "class_names": class_names,
        "splits": tables["split_summary"].to_dict("records"),
        "subject_overlap": {
            "num_subjects": int(len(overlap_df)),
            "num_subjects_in_multiple_splits": int((overlap_df["num_splits"] > 1).sum()) if not overlap_df.empty else 0,
            "subjects_in_multiple_splits": overlap_df.loc[overlap_df["num_splits"] > 1, "subject"].tolist() if not overlap_df.empty else [],
            "pairwise_subject_overlap": {
                f"{left}_{right}": int(((overlap_df[f"{left}_samples"] > 0) & (overlap_df[f"{right}_samples"] > 0)).sum())
                if not overlap_df.empty else 0
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            },
        },
    }
    if cli.include_effective_train and not validation_enabled:
        effective_records = records_by_split.get("train", []) + records_by_split.get("val", [])
        effective_tables = build_tables({"train": effective_records}, class_names)
        summary_payload["effective_train"] = effective_tables["split_summary"].to_dict("records")
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "distribution_report.md", dataset_id, cache_dir, run_params_path, split_summary_path, validation_enabled, class_names, effective_max_classes, resolved_names, tables, records_by_split, cli.include_effective_train)

    print(f"Splits: {', '.join(f'{split}={len(records_by_split.get(split, []))}' for split in LOGICAL_SPLITS)}")
    print(f"Classes: {len(class_names)}; subjects: {len(overlap_df)}")
    print(f"Saved report to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
