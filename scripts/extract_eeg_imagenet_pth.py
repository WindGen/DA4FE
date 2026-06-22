from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


REQUIRED_SAMPLE_KEYS = {"subject", "label", "image"}
EEG_KEY_CANDIDATES = ("eeg_data", "eeg")
DEFAULT_MAPPING_FILENAME = "index_name_mappings.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract EEG arrays from EEG-ImageNet PTH files into "
            "pth_name/class/subject/subject_class_image.npy folders and save "
            "label/image index mappings."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("E:\TeCh-improve\EEG-ImageNet/eeg_5_95_std.pth"),
        help="A directory that contains .pth files or a single .pth file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:\TeCh-improve\EEG-ImageNet"),# / "extracted_from_pth",
        help="Directory where extracted .npy files will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pth",
        help="Glob pattern used when --input points to a directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to extract from each PTH file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files instead of skipping them.",
    )
    parser.add_argument(
        "--mapping-name",
        default=DEFAULT_MAPPING_FILENAME,
        help="Filename used for the saved label/image mapping JSON.",
    )
    return parser.parse_args()


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to load .pth files. "
            "Install dependencies first, for example: pip install -r requirements.txt"
        )


def resolve_pth_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pth":
            raise ValueError(f"Expected a .pth file, got: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pth_files = sorted(input_path.glob(pattern))
    if not pth_files:
        raise FileNotFoundError(
            f"No .pth files found under {input_path} with pattern {pattern!r}"
        )
    return pth_files


def torch_load(path: Path):
    require_torch()
    kwargs = {"map_location": "cpu"}

    for extra_kwargs in (
        {"mmap": True, "weights_only": False},
        {"mmap": True},
        {},
    ):
        try:
            return torch.load(path, **kwargs, **extra_kwargs)
        except TypeError:
            continue
        except RuntimeError as exc:
            message = str(exc)
            if "mmap can only be used with files saved with" in message:
                continue
            raise

    return torch.load(path, **kwargs)


def resolve_dataset(payload, source: Path) -> list[dict]:
    if isinstance(payload, dict) and "dataset" in payload:
        dataset = payload["dataset"]
    else:
        dataset = payload

    if not isinstance(dataset, list):
        raise TypeError(
            f"Unsupported payload type in {source}: expected list or dict['dataset'], "
            f"got {type(dataset)!r}"
        )
    return dataset


def has_supported_dataset(payload) -> bool:
    try:
        dataset = resolve_dataset(payload, Path("<memory>"))
    except Exception:
        return False

    if not dataset:
        return False

    first = dataset[0]
    if not isinstance(first, dict):
        return False

    if not REQUIRED_SAMPLE_KEYS.issubset(first.keys()):
        return False

    return any(key in first for key in EEG_KEY_CANDIDATES)


def sanitize_part(value) -> str:
    text = str(value).strip()
    for ch in "\\/:*?\"<>|":
        text = text.replace(ch, "_")
    return text or "unknown"


def subject_to_text(value) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return sanitize_part(value)


def image_to_tag(value) -> str:
    return sanitize_part(Path(str(value)).stem)


def eeg_to_numpy(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def iter_samples(dataset: list[dict], limit: int | None, desc: str) -> Iterable[dict]:
    samples = dataset if limit is None else dataset[:limit]
    if tqdm is None:
        return samples
    return tqdm(samples, desc=desc, unit="sample")


def maybe_item(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def normalize_name_list(values) -> list[str]:
    if not isinstance(values, list):
        return []
    return [sanitize_part(value) for value in values]


def build_mapping_payload(payload) -> dict:
    labels = normalize_name_list(payload.get("labels", [])) if isinstance(payload, dict) else []
    images = normalize_name_list(payload.get("images", [])) if isinstance(payload, dict) else []
    return {
        "labels": {str(index): name for index, name in enumerate(labels)},
        "images": {str(index): name for index, name in enumerate(images)},
    }


def resolve_indexed_name(value, names: list[str], field_name: str) -> str:
    raw_value = maybe_item(value)
    if isinstance(raw_value, (int, np.integer)):
        index = int(raw_value)
        if not names:
            raise KeyError(
                f"Sample uses {field_name} index {index}, but the PTH payload does not "
                f"provide a top-level {field_name}s list."
            )
        if index < 0 or index >= len(names):
            raise IndexError(
                f"{field_name} index {index} is out of range for names list of "
                f"length {len(names)}."
            )
        return names[index]
    return sanitize_part(raw_value)


def resolve_eeg_key(sample: dict, source: Path, sample_index: int) -> str:
    for key in EEG_KEY_CANDIDATES:
        if key in sample:
            return key
    raise KeyError(
        f"Sample #{sample_index} in {source} is missing EEG data. "
        f"Expected one of {EEG_KEY_CANDIDATES}."
    )


def write_mapping_file(mapping_path: Path, payload: dict, source_name: str) -> None:
    mapping_data = {
        "source_pth": source_name,
        "labels": payload["labels"],
        "images": payload["images"],
    }
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(mapping_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_single_pth(
    pth_path: Path,
    output_root: Path,
    overwrite: bool,
    limit: int | None,
    mapping_name: str,
) -> tuple[int, int]:
    payload = torch_load(pth_path)
    if not has_supported_dataset(payload):
        raise ValueError(
            f"{pth_path.name} does not contain an extractable EEG dataset. "
            "This often happens for split/index files such as block_splits_*.pth."
        )

    dataset = resolve_dataset(payload, pth_path)
    mapping_payload = build_mapping_payload(payload)
    label_names = list(mapping_payload["labels"].values())
    image_names = list(mapping_payload["images"].values())

    file_output_root = output_root / pth_path.stem
    mapping_path = file_output_root / mapping_name
    write_mapping_file(mapping_path, mapping_payload, pth_path.name)

    saved = 0
    skipped = 0

    for index, sample in enumerate(iter_samples(dataset, limit, pth_path.name)):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample #{index} in {pth_path} is not a dict: {type(sample)!r}"
            )

        missing_keys = REQUIRED_SAMPLE_KEYS - sample.keys()
        if missing_keys:
            raise KeyError(
                f"Sample #{index} in {pth_path} is missing keys: {sorted(missing_keys)}"
            )

        eeg_key = resolve_eeg_key(sample, pth_path, index)
        class_label = resolve_indexed_name(sample["label"], label_names, "label")
        image_name = resolve_indexed_name(sample["image"], image_names, "image")
        subject_id = subject_to_text(sample["subject"])
        image_tag = image_to_tag(image_name)

        out_dir = file_output_root / class_label / subject_id
        out_path = out_dir / f"{subject_id}_{class_label}_{image_tag}.npy"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        eeg_array = eeg_to_numpy(sample[eeg_key])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, eeg_array)
        saved += 1

    del dataset
    del payload
    gc.collect()
    return saved, skipped


def main() -> None:
    args = parse_args()
    pth_files = resolve_pth_files(args.input, args.pattern)
    args.output.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0
    skipped_files: list[str] = []

    print(f"Found {len(pth_files)} PTH file(s).")
    print(f"Output root: {args.output.resolve()}")

    for pth_path in pth_files:
        print(f"Processing: {pth_path}")
        try:
            saved, skipped = extract_single_pth(
                pth_path=pth_path,
                output_root=args.output,
                overwrite=args.overwrite,
                limit=args.limit,
                mapping_name=args.mapping_name,
            )
        except ValueError as exc:
            print(f"Skipping {pth_path.name}: {exc}")
            skipped_files.append(pth_path.name)
            continue

        total_saved += saved
        total_skipped += skipped
        print(f"Finished {pth_path.name}: saved={saved}, skipped={skipped}")

    print(
        "All done. "
        f"saved={total_saved}, skipped={total_skipped}, output={args.output.resolve()}"
    )
    if skipped_files:
        print(f"Skipped non-dataset PTH files: {', '.join(skipped_files)}")


if __name__ == "__main__":
    main()
