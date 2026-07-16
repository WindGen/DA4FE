from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "picture"

METRIC_ALIASES = {
    "eval_loss": "loss",
    "train_step_loss": "opt_loss",
    "step_loss": "opt_loss",
    "top1": "top1_accuracy",
    "top3": "top3_accuracy",
    "top5": "top5_accuracy",
    "top10": "top10_accuracy",
}

METRIC_LABELS = {
    "loss": "Evaluation Loss",
    "opt_loss": "Train Step Loss",
    "accuracy": "Accuracy",
    "top1_accuracy": "Top-1 Accuracy",
    "top3_accuracy": "Top-3 Accuracy",
    "top5_accuracy": "Top-5 Accuracy",
    "top10_accuracy": "Top-10 Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Macro F1",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "learning_rate": "Learning Rate",
    "epoch_time_sec": "Epoch Time (s)",
}

METRICS_LOWER_IS_BETTER = {"loss", "opt_loss", "epoch_time_sec"}


@dataclass
class CVPRPlotStyle:
    """Compact plotting defaults tuned for paper-ready training curves."""

    dpi: int = 300
    figure_width: float = 7.2
    subplot_height: float = 2.6
    ncols: int = 2
    line_width: float = 2.15
    marker_size: float = 4.6
    marker_stride: int = 6
    best_marker_size: float = 52.0
    grid_alpha: float = 0.22
    grid_line_width: float = 0.7
    tick_size: float = 9.0
    label_size: float = 10.5
    title_size: float = 11.5
    legend_size: float = 9.0
    title_pad: float = 10.0
    font_family: str = "DejaVu Sans"
    face_color: str = "#FFFFFF"
    grid_color: str = "#CBD5E1"
    spine_color: str = "#334155"
    split_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "#4C78A8",
            "val": "#F58518",
            "test": "#54A24B",
            "val_final": "#E45756",
            "test_final": "#B279A2",
        }
    )
    split_markers: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "o",
            "val": "s",
            "test": "^",
            "val_final": "D",
            "test_final": "P",
        }
    )


def _normalize_metric_name(metric_name: str) -> str:
    metric_key = metric_name.strip().lower()
    return METRIC_ALIASES.get(metric_key, metric_key)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _empty_metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_id",
            "epoch",
            "epoch_num",
            "split",
            "loss",
            "opt_loss",
            "accuracy",
            "top1_accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "top10_accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "learning_rate",
            "epoch_time_sec",
            "steps",
        ]
    )


def load_training_log(log_path: str | Path) -> pd.DataFrame:
    """
    Load a TeCh training log.

    Supported formats:
    1. `metrics.csv` produced by the current training pipeline.
    2. Historical plain-text `.log` files printed to terminal redirection.

    Returns a tidy DataFrame with one row per epoch/split pair.
    """

    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if log_path.suffix.lower() == ".csv":
        frame = pd.read_csv(log_path)
        if "epoch" not in frame.columns or "split" not in frame.columns:
            raise ValueError(f"CSV log does not look like a metrics.csv file: {log_path}")
        frame = frame.copy()
        if "run_id" not in frame.columns:
            frame["run_id"] = 0
        frame["epoch_num"] = _safe_numeric(frame["epoch"])
        for column in frame.columns:
            if column not in {"epoch", "split"}:
                frame[column] = _safe_numeric(frame[column])
        if frame.empty:
            raise ValueError(
                f"No metric rows found in CSV log: {log_path}. "
                "Please confirm training has written at least one epoch to metrics.csv."
            )
        if "run_id" in frame.columns:
            frame["run_id"] = frame["run_id"].fillna(0).astype(int)
        return frame

    return _parse_plain_text_log(log_path)


def _parse_plain_text_log(log_path: Path) -> pd.DataFrame:
    metric_pattern = re.compile(
        r"Loss:\s*(?P<loss>[-+0-9.eE]+),\s*"
        r"Accuracy:\s*(?P<accuracy>[-+0-9.eE]+),\s*"
        r"(?:Top1:\s*(?P<top1>[-+0-9.eE]+),\s*)?"
        r"(?:Top3:\s*(?P<top3>[-+0-9.eE]+),\s*)?"
        r"(?:Top5:\s*(?P<top5>[-+0-9.eE]+),\s*)?"
        r"(?:Top10:\s*(?P<top10>[-+0-9.eE]+),\s*)?"
        r"Precision:\s*(?P<precision>[-+0-9.eE]+),\s*"
        r"Recall:\s*(?P<recall>[-+0-9.eE]+)\s*,?\s*"
        r"F1:\s*(?P<f1>[-+0-9.eE]+),\s*"
        r"AUROC:\s*(?P<auroc>[-+0-9.eE]+),\s*"
        r"AUPRC:\s*(?P<auprc>[-+0-9.eE]+)"
    )
    epoch_time_pattern = re.compile(r"Epoch:\s*(\d+)\s+cost time:\s*([-+0-9.eE]+)")
    train_step_pattern = re.compile(
        r"Epoch:\s*(\d+),\s*Steps:\s*(\d+),\s*\|\s*Train(?:\s+Step)?\s+Loss:\s*([-+0-9.eE]+)"
    )
    split_patterns = {
        "train": re.compile(r"^Train results ---\s*(.+)$"),
        "val": re.compile(r"^Validation results ---\s*(.+)$"),
        "test": re.compile(r"^Test results ---\s*(.+)$"),
        "val_final": re.compile(r"^Final validation results ---\s*(.+)$"),
        "test_final": re.compile(r"^Final test results ---\s*(.+)$"),
    }

    rows: List[Dict[str, float | int | str]] = []
    epoch_to_context: Dict[Tuple[int, int], Dict[str, float | int | None]] = {}
    current_epoch: Optional[int] = None
    current_run_id = 0
    last_epoch_seen: Optional[int] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            epoch_time_match = epoch_time_pattern.search(line)
            if epoch_time_match:
                next_epoch = int(epoch_time_match.group(1))
                if last_epoch_seen is not None and next_epoch < last_epoch_seen:
                    current_run_id += 1
                current_epoch = next_epoch
                last_epoch_seen = next_epoch
                epoch_key = (current_run_id, current_epoch)
                epoch_to_context.setdefault(epoch_key, {})
                epoch_to_context[epoch_key]["epoch_time_sec"] = float(
                    epoch_time_match.group(2)
                )
                continue

            train_step_match = train_step_pattern.search(line)
            if train_step_match:
                next_epoch = int(train_step_match.group(1))
                if last_epoch_seen is not None and next_epoch < last_epoch_seen:
                    current_run_id += 1
                current_epoch = next_epoch
                last_epoch_seen = next_epoch
                epoch_key = (current_run_id, current_epoch)
                epoch_to_context.setdefault(epoch_key, {})
                epoch_to_context[epoch_key]["steps"] = int(train_step_match.group(2))
                epoch_to_context[epoch_key]["opt_loss"] = float(train_step_match.group(3))
                continue

            matched_split = None
            matched_payload = None
            for split_name, split_pattern in split_patterns.items():
                split_match = split_pattern.search(line)
                if split_match:
                    matched_split = split_name
                    matched_payload = split_match.group(1)
                    break

            if matched_split is None or matched_payload is None:
                continue

            metrics_match = metric_pattern.search(matched_payload)
            if metrics_match is None:
                continue

            if matched_split.endswith("_final"):
                epoch_value: str | int = "final"
                epoch_num = math.nan
                context: Dict[str, float | int | None] = {}
            else:
                if current_epoch is None:
                    continue
                epoch_value = current_epoch
                epoch_num = float(current_epoch)
                context = epoch_to_context.get((current_run_id, current_epoch), {})

            row = {
                "run_id": current_run_id,
                "epoch": epoch_value,
                "epoch_num": epoch_num,
                "split": matched_split,
                "loss": float(metrics_match.group("loss")),
                "opt_loss": context.get("opt_loss", math.nan) if matched_split == "train" else math.nan,
                "accuracy": float(metrics_match.group("accuracy")),
                "top1_accuracy": float(metrics_match.group("top1") or metrics_match.group("accuracy")),
                "top3_accuracy": float(metrics_match.group("top3")) if metrics_match.group("top3") is not None else math.nan,
                "top5_accuracy": float(metrics_match.group("top5")) if metrics_match.group("top5") is not None else math.nan,
                "top10_accuracy": float(metrics_match.group("top10")) if metrics_match.group("top10") is not None else math.nan,
                "precision": float(metrics_match.group("precision")),
                "recall": float(metrics_match.group("recall")),
                "f1": float(metrics_match.group("f1")),
                "auroc": float(metrics_match.group("auroc")),
                "auprc": float(metrics_match.group("auprc")),
                "learning_rate": math.nan,
                "epoch_time_sec": context.get("epoch_time_sec", math.nan),
                "steps": context.get("steps", math.nan),
            }
            rows.append(row)

    if not rows:
        raise ValueError(f"Could not parse any metric rows from log: {log_path}")

    frame = pd.DataFrame(rows)
    return frame.sort_values(["run_id", "epoch_num", "split"], na_position="last").reset_index(drop=True)


def list_available_metrics(frame: pd.DataFrame) -> List[str]:
    available = []
    for metric_name in METRIC_LABELS:
        metric_key = _normalize_metric_name(metric_name)
        if metric_key in frame.columns and frame[metric_key].notna().any():
            available.append(metric_key)
    return sorted(set(available), key=available.index)


def plot_training_curves(
    log_path: str | Path,
    metrics: Sequence[str] = ("opt_loss", "loss", "accuracy", "f1"),
    splits: Sequence[str] = ("train", "val"),
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_name: Optional[str] = None,
    figure_title: Optional[str] = None,
    style: Optional[CVPRPlotStyle] = None,
    file_formats: Sequence[str] = ("png", "pdf"),
    smooth_window: int = 0,
    run_id: int | str = "last",
    include_final_metrics: bool = False,
    mark_best: bool = True,
    best_metric: str = "f1",
    best_split: str = "val",
    y_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    legend: bool = True,
) -> List[Path]:
    """
    Plot paper-style training curves from a TeCh log.

    Parameters
    ----------
    log_path:
        Path to `metrics.csv` or a historical `.log` file.
    metrics:
        Which metrics to draw. Common choices:
        `loss`, `opt_loss`, `accuracy`, `precision`, `recall`, `f1`,
        `auroc`, `auprc`, `learning_rate`, `epoch_time_sec`.
    splits:
        Which splits to show. Usually `("train", "val")` or
        `("train", "val", "test")`.
    output_dir:
        Directory for saved figures. Defaults to `<repo>/picture`.
    output_name:
        Base file name without extension. If omitted, it is derived from the
        log file name and selected metrics.
    figure_title:
        Optional title shown above the full figure.
    style:
        `CVPRPlotStyle` instance for later visual tweaking.
    file_formats:
        Save formats, e.g. `("png", "pdf")`.
    smooth_window:
        Rolling window size for optional curve smoothing. `0` or `1` disables it.
    run_id:
        Which training run to draw when one `.log` file contains multiple appended
        runs. Use `"last"` (default), `"all"`, or a specific integer run id.
    include_final_metrics:
        Whether to include `val_final` / `test_final` rows if present.
    mark_best:
        Whether to mark the best epoch on the chosen `best_metric`.
    best_metric:
        Metric used to locate the best epoch marker.
    best_split:
        Split used to locate the best epoch marker.
    y_limits:
        Optional per-metric bounds, e.g. `{"f1": (0.0, 1.0)}`.
    legend:
        Whether to draw a legend on the first subplot.
    """

    style = style or CVPRPlotStyle()
    frame = load_training_log(log_path)

    if frame.empty:
        raise ValueError(
            f"Loaded log is empty: {log_path}. "
            "Please confirm metrics.csv contains epoch-level train/val rows."
        )

    if "run_id" not in frame.columns:
        frame["run_id"] = 0
    frame["run_id"] = _safe_numeric(frame["run_id"]).fillna(0)

    if run_id != "all":
        if run_id == "last":
            run_id_max = frame["run_id"].max()
            if pd.isna(run_id_max):
                raise ValueError(
                    f"Could not infer a valid run_id from log: {log_path}. "
                    "Please confirm metrics.csv contains at least one non-empty data row."
                )
            selected_run_id = int(run_id_max)
        else:
            selected_run_id = int(run_id)
        frame = frame[frame["run_id"] == selected_run_id].copy()
        if frame.empty:
            raise ValueError(f"Requested run_id={selected_run_id} is not present in log.")

    metric_keys = [_normalize_metric_name(metric_name) for metric_name in metrics]
    missing_metrics = [metric for metric in metric_keys if metric not in frame.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in log: {missing_metrics}")

    requested_splits = list(splits)
    if include_final_metrics:
        for final_split in ("val_final", "test_final"):
            if final_split not in requested_splits:
                requested_splits.append(final_split)

    plot_frame = frame.copy()
    plot_frame = plot_frame[plot_frame["split"].isin(requested_splits)]
    plot_frame = plot_frame[plot_frame["epoch_num"].notna()]

    if plot_frame.empty:
        raise ValueError(
            "No epoch-wise rows are available for the requested splits. "
            "Try using train/val/test instead of *_final only."
        )

    for metric_name in metric_keys:
        plot_frame[metric_name] = _safe_numeric(plot_frame[metric_name])

    plot_frame["epoch_num"] = _safe_numeric(plot_frame["epoch_num"])
    plot_frame = plot_frame.sort_values(["split", "epoch_num"]).reset_index(drop=True)

    if smooth_window and smooth_window > 1:
        for metric_name in metric_keys:
            plot_frame[metric_name] = (
                plot_frame.groupby("split")[metric_name]
                .transform(lambda values: values.rolling(smooth_window, min_periods=1).mean())
            )

    available_splits = [
        split_name for split_name in requested_splits if split_name in plot_frame["split"].unique()
    ]
    if not available_splits:
        raise ValueError(f"Requested splits are not present in log: {requested_splits}")

    n_metrics = len(metric_keys)
    ncols = max(1, min(style.ncols, n_metrics))
    nrows = math.ceil(n_metrics / ncols)
    figsize = (style.figure_width, style.subplot_height * nrows)

    rc_params = {
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
        "font.family": style.font_family,
        "font.size": style.tick_size,
        "axes.labelsize": style.label_size,
        "axes.titlesize": style.title_size,
        "xtick.labelsize": style.tick_size,
        "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.legend_size,
        "axes.facecolor": style.face_color,
        "figure.facecolor": style.face_color,
        "axes.grid": True,
        "grid.color": style.grid_color,
        "grid.alpha": style.grid_alpha,
        "grid.linewidth": style.grid_line_width,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
            constrained_layout=False,
        )
        axes_flat = axes.flatten()

        for axis_index, metric_name in enumerate(metric_keys):
            axis = axes_flat[axis_index]
            metric_frame = plot_frame[["epoch_num", "split", metric_name]].dropna(subset=[metric_name])

            for split_name in available_splits:
                split_frame = metric_frame[metric_frame["split"] == split_name]
                if split_frame.empty:
                    continue

                axis.plot(
                    split_frame["epoch_num"].to_numpy(),
                    split_frame[metric_name].to_numpy(),
                    label=split_name,
                    color=style.split_colors.get(split_name, "#4C78A8"),
                    linewidth=style.line_width,
                    marker=style.split_markers.get(split_name, "o"),
                    markersize=style.marker_size,
                    markevery=style.marker_stride,
                    markerfacecolor=style.face_color,
                    markeredgewidth=1.1,
                    alpha=0.98,
                )

            axis.set_title(METRIC_LABELS.get(metric_name, metric_name.upper()), pad=style.title_pad)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(METRIC_LABELS.get(metric_name, metric_name.upper()))
            axis.set_xlim(left=1)

            if y_limits and metric_name in y_limits:
                axis.set_ylim(*y_limits[metric_name])

            for spine_name in ("left", "bottom"):
                axis.spines[spine_name].set_color(style.spine_color)
                axis.spines[spine_name].set_linewidth(1.0)

            if legend and axis_index == 0:
                axis.legend(loc="best", frameon=False, ncol=min(3, len(available_splits)))

            if mark_best and metric_name == _normalize_metric_name(best_metric):
                best_frame = metric_frame[metric_frame["split"] == best_split]
                if not best_frame.empty:
                    comparator = best_frame[metric_name].idxmin()
                    if metric_name not in METRICS_LOWER_IS_BETTER:
                        comparator = best_frame[metric_name].idxmax()
                    best_row = best_frame.loc[comparator]
                    axis.scatter(
                        [best_row["epoch_num"]],
                        [best_row[metric_name]],
                        s=style.best_marker_size,
                        color="#111827",
                        zorder=5,
                        label=None,
                    )
                    axis.annotate(
                        f"best {best_split}: {best_row[metric_name]:.4f}",
                        xy=(best_row["epoch_num"], best_row[metric_name]),
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=style.legend_size,
                        color="#111827",
                    )

        for axis in axes_flat[n_metrics:]:
            axis.remove()

        if figure_title is None:
            figure_title = Path(log_path).stem.replace("_", " ")
        if figure_title:
            fig.suptitle(figure_title, fontsize=style.title_size, y=0.995)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_name is None:
            metric_suffix = "_".join(metric_keys)
            if run_id == "all":
                output_name = f"{Path(log_path).stem}_{metric_suffix}_all_runs"
            else:
                run_suffix = "last" if run_id == "last" else f"run{int(run_id)}"
                output_name = f"{Path(log_path).stem}_{metric_suffix}_{run_suffix}"

        saved_paths: List[Path] = []
        for file_format in file_formats:
            normalized_format = file_format.lower().lstrip(".")
            save_path = output_dir / f"{output_name}.{normalized_format}"
            fig.savefig(save_path, bbox_inches="tight", facecolor=style.face_color)
            saved_paths.append(save_path)

        plt.close(fig)

    return saved_paths


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot CVPR-style training curves from TeCh metrics.csv or .log files."
    )
    parser.add_argument("--log_path", type=str, required=True, help="Path to metrics.csv or .log.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["opt_loss", "loss", "accuracy", "f1"],
        help="Metrics to draw. Example: --metrics f1 accuracy auroc",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to draw. Example: --splits train val test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved figures. Default: <repo>/picture",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional base name for the saved figure.",
    )
    parser.add_argument(
        "--figure_title",
        type=str,
        default=None,
        help="Optional figure title. Default uses the log file stem.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=0,
        help="Rolling average window for smoothing. 0 or 1 disables smoothing.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="last",
        help='Which run to draw if one .log contains multiple appended runs. Use "last", "all", or an integer id.',
    )
    parser.add_argument(
        "--include_final_metrics",
        action="store_true",
        help="Include val_final / test_final rows if they exist.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="f1",
        help="Metric used for the best-epoch marker.",
    )
    parser.add_argument(
        "--best_split",
        type=str,
        default="val",
        help="Split used for the best-epoch marker.",
    )
    parser.add_argument(
        "--no_best_marker",
        action="store_true",
        help="Disable the best-epoch marker.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Save formats. Example: --formats png pdf svg",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    saved_paths = plot_training_curves(
        log_path=args.log_path,
        metrics=args.metrics,
        splits=args.splits,
        output_dir=args.output_dir,
        output_name=args.output_name,
        figure_title=args.figure_title,
        file_formats=args.formats,
        smooth_window=args.smooth_window,
        run_id=args.run_id,
        include_final_metrics=args.include_final_metrics,
        mark_best=not args.no_best_marker,
        best_metric=args.best_metric,
        best_split=args.best_split,
    )
    print("Saved figures:")
    for save_path in saved_paths:
        print(save_path)


if __name__ == "__main__":
    main()
