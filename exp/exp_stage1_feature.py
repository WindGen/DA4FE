from copy import deepcopy
import csv
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from pytorch_metric_learning import losses as pml_losses
    from pytorch_metric_learning import miners as pml_miners
except ImportError:  # Optional until a Multi-Similarity mode is selected.
    pml_losses = None
    pml_miners = None

from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from exp.exp_basic import Exp_Basic
from utils.checkpointing import (
    build_checkpoint_payload,
    extract_model_state,
    load_checkpoint_object,
    load_model_state,
    unwrap_model,
)
from utils.stage_metrics import compute_kmeans_score
from utils.tools import adjust_learning_rate
from models.eeg_clip_alignment import EEGImageProjection

warnings.filterwarnings("ignore")


class ArcMarginHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m)
        self.mm = np.sin(np.pi - self.m) * self.m

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class CosMarginHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).float()
        output = cosine - one_hot * self.m
        return output * self.s


class Stage1FeatureModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_class, args, clip_feature_dim=0, resnet_feature_dim=0):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.loss_mode = getattr(args, "stage1_loss", "triplet").lower()
        l2_normalize = getattr(args, "stage1_l2_normalize", True)
        if isinstance(l2_normalize, str):
            l2_normalize = l2_normalize.strip().lower() in {"true", "1", "yes", "y", "on"}
        self.l2_normalize_features = bool(l2_normalize)
        self.margin_head = None
        self.clip_projection = None
        if clip_feature_dim > 0:
            self.clip_projection = EEGImageProjection(
                input_dim=feature_dim,
                output_dim=clip_feature_dim,
                hidden_dim=getattr(args, "stage1_clip_projection_hidden_dim", 1024),
                dropout=getattr(args, "stage1_clip_projection_dropout", 0.1),
            )
        self.resnet_projection = None
        if resnet_feature_dim > 0:
            self.resnet_projection = EEGImageProjection(
                input_dim=feature_dim,
                output_dim=resnet_feature_dim,
                hidden_dim=getattr(args, "stage1_resnet_projection_hidden_dim", 1024),
                dropout=getattr(args, "stage1_resnet_projection_dropout", 0.1),
            )

        if self.loss_mode in {"arcface", "arcface_triplet"}:
            self.margin_head = ArcMarginHead(
                in_features=feature_dim,
                out_features=num_class,
                s=getattr(args, "stage1_arcface_s", 30.0),
                m=getattr(args, "stage1_arcface_m", 0.5),
            )
        elif self.loss_mode in {"cosface", "cosface_triplet"}:
            self.margin_head = CosMarginHead(
                in_features=feature_dim,
                out_features=num_class,
                s=getattr(args, "stage1_cosface_s", 30.0),
                m=getattr(args, "stage1_cosface_m", 0.35),
            )

    def forward(self, x_enc, return_features=False):
        logits, fused = self.backbone(x_enc, return_features=True)
        if return_features:
            if self.l2_normalize_features:
                fused = F.normalize(fused, p=2, dim=-1)
            return logits, fused
        return logits


class TripletMarginMiner:
    def __init__(self, margin=0.2, miner_type="semihard"):
        self.margin = float(margin)
        self.miner_type = miner_type
        if self.miner_type not in {"semihard", "batch_hard"}:
            raise ValueError(f"Unsupported stage1 miner type: {self.miner_type}")

    def __call__(self, embeddings, labels):
        if embeddings.ndim != 2:
            raise ValueError("Triplet embeddings must be a 2D tensor")

        labels = labels.reshape(-1)
        batch_size = labels.size(0)
        if batch_size < 2:
            return None

        with torch.no_grad():
            distances = torch.cdist(embeddings.detach(), embeddings.detach(), p=2)
            same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
            diag_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
            positive_mask = same_label & ~diag_mask
            negative_mask = ~same_label

            anchors = []
            positives = []
            negatives = []

            for anchor_idx in range(batch_size):
                positive_idx = torch.nonzero(
                    positive_mask[anchor_idx], as_tuple=False
                ).flatten()
                negative_idx = torch.nonzero(
                    negative_mask[anchor_idx], as_tuple=False
                ).flatten()

                if positive_idx.numel() == 0 or negative_idx.numel() == 0:
                    continue

                positive_distances = distances[anchor_idx, positive_idx]
                hardest_positive = positive_idx[positive_distances.argmax()]
                hardest_negative = negative_idx[
                    distances[anchor_idx, negative_idx].argmin()
                ]

                if self.miner_type == "batch_hard":
                    anchors.append(anchor_idx)
                    positives.append(int(hardest_positive))
                    negatives.append(int(hardest_negative))
                    continue

                selected = False
                sorted_positive_positions = torch.argsort(
                    positive_distances, descending=True
                )
                for pos_position in sorted_positive_positions:
                    candidate_positive = positive_idx[pos_position]
                    d_ap = distances[anchor_idx, candidate_positive]
                    negative_distances = distances[anchor_idx, negative_idx]
                    semi_hard_mask = (
                        (negative_distances > d_ap)
                        & (negative_distances < d_ap + self.margin)
                    )
                    if semi_hard_mask.any():
                        semi_hard_negatives = negative_idx[semi_hard_mask]
                        selected_negative = semi_hard_negatives[
                            distances[anchor_idx, semi_hard_negatives].argmin()
                        ]
                        anchors.append(anchor_idx)
                        positives.append(int(candidate_positive))
                        negatives.append(int(selected_negative))
                        selected = True
                        break

                if not selected:
                    anchors.append(anchor_idx)
                    positives.append(int(hardest_positive))
                    negatives.append(int(hardest_negative))

        if not anchors:
            return None

        return (
            torch.tensor(anchors, device=embeddings.device, dtype=torch.long),
            torch.tensor(positives, device=embeddings.device, dtype=torch.long),
            torch.tensor(negatives, device=embeddings.device, dtype=torch.long),
        )


class ClassBalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size, samples_per_class=2, drop_last=False, seed=42):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        if samples_per_class < 2:
            raise ValueError(
                "stage1_samples_per_class must be >= 2 because metric learning "
                "requires at least two samples from the same class in a batch."
            )
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        self.seed = seed
        self._iteration = 0

        self.class_to_indices = {}
        for index, label in enumerate(self.labels):
            self.class_to_indices.setdefault(int(label), []).append(index)
        self.available_classes = [
            class_id
            for class_id, indices in self.class_to_indices.items()
            if len(indices) >= self.samples_per_class
        ]
        self.classes_per_batch = max(1, self.batch_size // self.samples_per_class)

        if not self.available_classes:
            raise ValueError(
                "ClassBalancedBatchSampler requires at least one class with "
                f"{self.samples_per_class} samples."
            )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        per_class_indices = {
            class_id: rng.permutation(indices).tolist()
            for class_id, indices in self.class_to_indices.items()
        }
        per_class_offsets = {class_id: 0 for class_id in self.class_to_indices}

        total_indices = len(self.labels)
        yielded = 0
        while yielded < total_indices:
            chosen_classes = rng.choice(
                self.available_classes,
                size=min(self.classes_per_batch, len(self.available_classes)),
                replace=False,
            )
            batch = []
            for class_id in chosen_classes:
                indices = per_class_indices[class_id]
                offset = per_class_offsets[class_id]
                if offset + self.samples_per_class > len(indices):
                    indices = rng.permutation(self.class_to_indices[class_id]).tolist()
                    per_class_indices[class_id] = indices
                    offset = 0
                selected = indices[offset : offset + self.samples_per_class]
                per_class_offsets[class_id] = offset + self.samples_per_class
                batch.extend(selected)

            if len(batch) < self.batch_size:
                remaining = self.batch_size - len(batch)
                extra_indices = rng.choice(
                    total_indices,
                    size=remaining,
                    replace=remaining > total_indices,
                ).tolist()
                batch.extend(extra_indices)

            batch = batch[: self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yielded += len(batch)
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return int(np.ceil(len(self.labels) / self.batch_size))


class MergedStage1Dataset(ConcatDataset):
    """Concatenate train and validation datasets while preserving metadata.

    The stage-1 sampler and model builder rely on ``y``/``num_class`` and the
    sequence metadata exposed by the individual dataset implementations.
    ``torch.utils.data.ConcatDataset`` only implements indexing, so this small
    adapter carries those attributes over to the merged training split.
    """

    def __init__(self, *datasets):
        datasets = tuple(dataset for dataset in datasets if dataset is not None)
        if not datasets:
            raise ValueError("At least one dataset is required")
        super().__init__(datasets)

        labels = []
        for dataset in datasets:
            if not hasattr(dataset, "y"):
                raise AttributeError("Stage-1 datasets must expose a `y` label array")
            labels.append(np.asarray(dataset.y, dtype=np.int64).reshape(-1))
        self.y = np.concatenate(labels, axis=0)

        first = datasets[0]
        self.num_class = int(
            max(
                getattr(dataset, "num_class", 0) for dataset in datasets
            )
        )
        if self.num_class <= 0:
            self.num_class = int(np.unique(self.y).size)
        for attribute in ("max_seq_len", "feature_dim", "class_names"):
            if hasattr(first, attribute):
                setattr(self, attribute, getattr(first, attribute))


class Stage1ImageFeatureDataset(Dataset):
    """Attach cached paired-image teacher embeddings to Stage-1 EEG samples."""

    def __init__(self, dataset, clip_features=None, resnet_features=None):
        self.dataset = dataset
        self.clip_features = None if clip_features is None else torch.as_tensor(clip_features, dtype=torch.float32)
        self.resnet_features = None if resnet_features is None else torch.as_tensor(resnet_features, dtype=torch.float32)
        if self.clip_features is not None and len(dataset) != len(self.clip_features):
            raise ValueError("CLIP feature count does not match the Stage-1 dataset")
        if self.resnet_features is not None and len(dataset) != len(self.resnet_features):
            raise ValueError("ResNet50 feature count does not match the Stage-1 dataset")
        self.y = np.asarray(dataset.y, dtype=np.int64)
        for attribute in ("num_class", "max_seq_len", "feature_dim", "class_names"):
            if hasattr(dataset, attribute):
                setattr(self, attribute, getattr(dataset, attribute))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        eeg, label = self.dataset[index]
        clip_features = None if self.clip_features is None else self.clip_features[index]
        resnet_features = None if self.resnet_features is None else self.resnet_features[index]
        return eeg, label, clip_features, resnet_features


def stage1_collate_fn(batch, max_len):
    """Collate either ordinary EEG pairs or EEG pairs with CLIP targets."""
    if len(batch[0]) == 2:
        return collate_fn(batch, max_len=max_len)
    eeg_batch, labels, padding_masks = collate_fn(
        [(sample[0], sample[1]) for sample in batch], max_len=max_len
    )
    clip_targets = torch.stack([sample[2] for sample in batch], dim=0) if batch[0][2] is not None else None
    resnet_targets = torch.stack([sample[3] for sample in batch], dim=0) if batch[0][3] is not None else None
    return eeg_batch, labels, padding_masks, clip_targets, resnet_targets


class Exp_Stage1_Feature(Exp_Basic):
    METRIC_FIELDNAMES = [
        "epoch",
        "split",
        "loss",
        "classification_loss",
        "metric_loss",
        "clip_loss",
        "resnet_loss",
        "kmeans_score",
        "mined_triplets",
        "learning_rate",
        "epoch_time_sec",
        "steps",
    ]

    TRIPLET_LOSS_MODES = {"triplet", "ce_triplet", "arcface_triplet", "cosface_triplet"}
    MULTI_SIMILARITY_LOSS_MODES = {"multi_similarity", "ce_multi_similarity"}
    METRIC_LOSS_MODES = TRIPLET_LOSS_MODES | MULTI_SIMILARITY_LOSS_MODES
    CLASSIFICATION_LOSS_MODES = {
        "ce",
        "ce_triplet",
        "arcface",
        "arcface_triplet",
        "cosface",
        "cosface_triplet",
        "ce_multi_similarity",
    }

    def __init__(self, args):
        self.loss_mode = getattr(args, "stage1_loss", "triplet").lower()
        self.use_validation = bool(getattr(args, "use_validation", True))
        self.triplet_miner = None
        self.triplet_criterion = None
        self.ms_miner = None
        self.ms_criterion = None
        self.classification_criterion = None
        self.clip_model = None
        self.clip_processor = None
        self.resnet_model = None
        self.resnet_processor = None
        self.clip_feature_dim = 0
        self.resnet_feature_dim = 0
        self._image_dataset_cache = {}
        self._image_feature_cache = {}
        self.clip_enabled = float(getattr(args, "stage1_clip_weight", 0.0)) > 0.0
        self.resnet_enabled = float(getattr(args, "stage1_resnet_weight", 0.0)) > 0.0
        self.image_alignment_enabled = self.clip_enabled or self.resnet_enabled
        super().__init__(args)
        self.metrics_log_path = None
        self.split_summary_path = None
        self.run_params_path = None

    def _dataset_seq_len(self, dataset):
        if hasattr(dataset, "max_seq_len"):
            return dataset.max_seq_len
        return dataset.X.shape[1]

    def _dataset_feature_dim(self, dataset):
        if hasattr(dataset, "feature_dim"):
            return dataset.feature_dim
        return dataset.X.shape[2]

    def _dataset_num_class(self, dataset):
        if hasattr(dataset, "num_class"):
            return dataset.num_class
        return len(np.unique(dataset.y))

    def _merge_datasets(self, *datasets):
        """Merge datasets for no-validation training while retaining metadata."""
        return MergedStage1Dataset(*datasets)

    def _dataset_summary(self, dataset, split_name):
        seq_len = self._dataset_seq_len(dataset)
        feature_dim = self._dataset_feature_dim(dataset)
        num_class = self._dataset_num_class(dataset)
        return (
            f"{split_name}: samples={len(dataset)}, seq_len={seq_len}, "
            f"feat_dim={feature_dim}, num_class={num_class}"
        )

    def _dataset_summary_payload(self, dataset, split_name):
        if dataset is None:
            return {
                "split": split_name,
                "samples": 0,
                "seq_len": None,
                "feat_dim": None,
                "num_class": 0,
            }
        return {
            "split": split_name,
            "samples": len(dataset),
            "seq_len": self._dataset_seq_len(dataset),
            "feat_dim": self._dataset_feature_dim(dataset),
            "num_class": self._dataset_num_class(dataset),
        }

    def _checkpoint_dir(self, setting):
        result_root = getattr(self.args, "result_dir", None)
        if result_root is None:
            # Keep generated weights outside the source tree by default.
            result_root = Path(__file__).resolve().parents[2] / "result"
        else:
            result_root = Path(result_root).expanduser()
        return result_root / self.args.model / "stage1" / setting

    def _log_dir(self, setting):
        log_root = getattr(self.args, "log_dir", None)
        if log_root is None:
            return self._checkpoint_dir(setting)
        return Path(log_root) / self.args.data / "stage1" / setting

    def _prepare_metric_logging(self, setting, train_data, vali_data, test_data):
        checkpoint_dir = self._checkpoint_dir(setting)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self._log_dir(setting)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_log_path = log_dir / "metrics.csv"
        self.split_summary_path = log_dir / "split_summary.json"
        self.run_params_path = checkpoint_dir / "run_params.json"

        with self.metrics_log_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.METRIC_FIELDNAMES)
            writer.writeheader()

        run_params_payload = {
            "stage": "stage1_feature_training",
            "run_directory_name": setting,
            "full_setting_name": getattr(self.args, "full_setting_name", setting),
            "run_started_at": getattr(self.args, "run_started_at", None),
            "validation_enabled": self.use_validation,
            "paths": {
                "checkpoint_dir": str(checkpoint_dir),
                "log_dir": str(log_dir),
            },
            "args": vars(deepcopy(self.args)),
        }
        self.run_params_path.write_text(
            json.dumps(run_params_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if log_dir != checkpoint_dir:
            (log_dir / "run_params.json").write_text(
                json.dumps(run_params_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        summary_payload = {
            "setting": setting,
            "full_setting_name": getattr(self.args, "full_setting_name", setting),
            "validation_enabled": self.use_validation,
            "args": vars(deepcopy(self.args)),
            "splits": {
                "train": self._dataset_summary_payload(train_data, "TRAIN"),
                "val": self._dataset_summary_payload(vali_data, "VAL"),
                "test": self._dataset_summary_payload(test_data, "TEST"),
            },
        }
        self.split_summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_metric_row(
        self,
        epoch,
        split,
        loss,
        classification_loss,
        metric_loss,
        clip_loss,
        resnet_loss,
        kmeans_score,
        mined_triplets=None,
        learning_rate=None,
        epoch_time=None,
        steps=None,
    ):
        if self.metrics_log_path is None:
            return

        row = {
            "epoch": epoch,
            "split": split,
            "loss": float(loss),
            "classification_loss": float(classification_loss),
            "metric_loss": float(metric_loss),
            "clip_loss": float(clip_loss),
            "resnet_loss": float(resnet_loss),
            "kmeans_score": float(kmeans_score),
            "mined_triplets": "" if mined_triplets is None else int(mined_triplets),
            "learning_rate": "" if learning_rate is None else float(learning_rate),
            "epoch_time_sec": "" if epoch_time is None else float(epoch_time),
            "steps": "" if steps is None else int(steps),
        }
        with self.metrics_log_path.open("a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.METRIC_FIELDNAMES)
            writer.writerow(row)

    def _feature_kmeans_clusters(self):
        requested_clusters = int(getattr(self.args, "stage1_kmeans_clusters", 0))
        return requested_clusters if requested_clusters > 0 else self.args.num_class

    def _infer_backbone_feature_dim(self, backbone):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.args.seq_len, self.args.enc_in)
            _, fused = backbone(dummy_input, return_features=True)
        return int(fused.shape[-1])

    def _build_model(self):
        test_data, _ = self._get_data(flag="TEST")
        self.args.seq_len = self._dataset_seq_len(test_data)
        self.args.pred_len = 0
        self.args.enc_in = self._dataset_feature_dim(test_data)
        self.args.num_class = self._dataset_num_class(test_data)

        backbone = self.model_dict[self.args.model].Model(self.args).float()
        feature_dim = self._infer_backbone_feature_dim(backbone)
        self.args.stage1_backbone_feature_dim = feature_dim
        self.clip_feature_dim = (
            int(getattr(self.args, "stage1_clip_feature_dim", 768)) if self.clip_enabled else 0
        )
        self.resnet_feature_dim = (
            int(getattr(self.args, "stage1_resnet_feature_dim", 2048)) if self.resnet_enabled else 0
        )

        model = Stage1FeatureModel(
            backbone=backbone,
            feature_dim=feature_dim,
            num_class=self.args.num_class,
            args=self.args,
            clip_feature_dim=self.clip_feature_dim,
            resnet_feature_dim=self.resnet_feature_dim,
        ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _resolve_teacher_source(self, explicit_path, local_name, hub_id):
        pretrained_root = Path(getattr(self.args, "pretrained_root", "../pretrained")).expanduser()
        if explicit_path:
            explicit_text = str(explicit_path)
            explicit_local = Path(explicit_text).expanduser()
            if explicit_local.exists():
                return str(explicit_local), True
            normalized = explicit_text.replace("\\", "/")
            is_absolute_path = explicit_local.is_absolute() or normalized.startswith(("./", "../", "/")) or (len(normalized) > 2 and normalized[1:3] == ":/")
            if is_absolute_path:
                raise FileNotFoundError(f"Pretrained image model path does not exist: {explicit_text}")
            return explicit_text, False
        local_default = pretrained_root / local_name
        if local_default.exists():
            return str(local_default), True
        return hub_id, False

    def _load_image_teachers(self, image_dataset):
        if not callable(getattr(image_dataset, "get_image", None)):
            raise ValueError(
                "Stage-1 image-feature alignment requires paired images from the "
                "Hugging Face EEG-ImageNet dataset. The selected dataset has no get_image(index)."
            )
        try:
            if self.clip_enabled and (self.clip_model is None or self.clip_processor is None):
                from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

                source, is_local = self._resolve_teacher_source(
                    getattr(self.args, "stage1_clip_model_path", None),
                    "clip-vit-large-patch14",
                    "openai/clip-vit-large-patch14",
                )
                cache_dir = getattr(self.args, "hf_cache_dir", None)
                local_files_only = is_local or not bool(getattr(self.args, "allow_hf_download", True))
                self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
                    source, cache_dir=cache_dir, local_files_only=local_files_only
                ).to(self.device).eval()
                self.clip_processor = CLIPImageProcessor.from_pretrained(
                    source, cache_dir=cache_dir, local_files_only=local_files_only
                )
                for parameter in self.clip_model.parameters():
                    parameter.requires_grad = False
                clip_dim = getattr(self.clip_model.config, "projection_dim", None)
                if clip_dim is None:
                    clip_dim = self.clip_model.visual_projection.out_features
                if int(clip_dim) != self.clip_feature_dim:
                    raise ValueError(
                        f"Configured stage1_clip_feature_dim={self.clip_feature_dim}, but the loaded CLIP model outputs {int(clip_dim)} dimensions."
                    )
                print(f"Stage-1 CLIP teacher: {source} (feature_dim={self.clip_feature_dim})")

            if self.resnet_enabled and (self.resnet_model is None or self.resnet_processor is None):
                from transformers import AutoImageProcessor, ResNetForImageClassification

                source, is_local = self._resolve_teacher_source(
                    getattr(self.args, "stage1_resnet_model_path", None),
                    "resnet-50",
                    "microsoft/resnet-50",
                )
                cache_dir = getattr(self.args, "hf_cache_dir", None)
                local_files_only = is_local or not bool(getattr(self.args, "allow_hf_download", True))
                self.resnet_model = ResNetForImageClassification.from_pretrained(
                    source, cache_dir=cache_dir, local_files_only=local_files_only
                ).to(self.device).eval()
                self.resnet_processor = AutoImageProcessor.from_pretrained(
                    source, cache_dir=cache_dir, local_files_only=local_files_only
                )
                for parameter in self.resnet_model.parameters():
                    parameter.requires_grad = False
                hidden_sizes = getattr(self.resnet_model.config, "hidden_sizes", None)
                resnet_dim = int(hidden_sizes[-1]) if hidden_sizes else self.resnet_model.classifier[-1].in_features
                if resnet_dim != self.resnet_feature_dim:
                    raise ValueError(
                        f"Configured stage1_resnet_feature_dim={self.resnet_feature_dim}, but the loaded ResNet model outputs {resnet_dim} dimensions."
                    )
                print(f"Stage-1 ResNet50 teacher: {source} (feature_dim={self.resnet_feature_dim})")
        except ImportError as exc:
            raise ImportError(
                "Stage-1 CLIP/ResNet image alignment requires transformers==4.44.2."
            ) from exc

    def _with_image_targets(self, dataset):
        if not self.image_alignment_enabled:
            return dataset
        cached = self._image_dataset_cache.get(id(dataset))
        if cached is not None:
            return cached
        if not callable(getattr(dataset, "get_image", None)):
            raise ValueError(
                "Stage-1 image-feature alignment is enabled, but this dataset has no paired image accessor."
            )

        underlying_dataset = getattr(dataset, "dataset", dataset)
        feature_key = (
            id(underlying_dataset),
            getattr(dataset, "dataset_id", None),
            getattr(dataset, "split_name", getattr(dataset, "flag", None)),
        )
        image_targets = self._image_feature_cache.get(feature_key)
        if image_targets is None:
            self._load_image_teachers(dataset)
            batch_size = max(1, int(getattr(self.args, "stage1_clip_batch_size", 32)))
            clip_batches = []
            resnet_batches = []
            for start in tqdm(
                range(0, len(dataset), batch_size),
                desc=f"Extract paired image features ({getattr(dataset, 'flag', 'split')})",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            ):
                images = [dataset.get_image(index) for index in range(start, min(start + batch_size, len(dataset)))]
                with torch.no_grad():
                    if self.clip_enabled:
                        clip_pixels = self.clip_processor(images=images, return_tensors="pt").pixel_values.to(self.device)
                        clip_features = self.clip_model(pixel_values=clip_pixels).image_embeds.float()
                        clip_batches.append(F.normalize(clip_features, p=2, dim=-1).cpu())
                    if self.resnet_enabled:
                        resnet_pixels = self.resnet_processor(images=images, return_tensors="pt").pixel_values.to(self.device)
                        resnet_output = self.resnet_model.resnet(pixel_values=resnet_pixels)
                        resnet_features = resnet_output.pooler_output.float().flatten(1)
                        resnet_batches.append(F.normalize(resnet_features, p=2, dim=-1).cpu())

            image_targets = {
                "clip": torch.cat(clip_batches, dim=0) if self.clip_enabled else None,
                "resnet": torch.cat(resnet_batches, dim=0) if self.resnet_enabled else None,
            }
            self._image_feature_cache[feature_key] = image_targets
        wrapped = Stage1ImageFeatureDataset(dataset, image_targets["clip"], image_targets["resnet"])
        self._image_dataset_cache[id(dataset)] = wrapped
        return wrapped

    def _release_image_teachers(self):
        if self.clip_model is not None:
            self.clip_model.to("cpu")
            self.clip_model = None
            self.clip_processor = None
        if self.resnet_model is not None:
            self.resnet_model.to("cpu")
            self.resnet_model = None
            self.resnet_processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_data(self, flag):
        random.seed(self.args.seed)
        return data_provider(self.args, flag)

    def _use_triplet_loss(self):
        return self.loss_mode in self.TRIPLET_LOSS_MODES

    def _use_multi_similarity_loss(self):
        return self.loss_mode in self.MULTI_SIMILARITY_LOSS_MODES

    def _use_metric_loss(self):
        return self.loss_mode in self.METRIC_LOSS_MODES

    def _use_classification_loss(self):
        return self.loss_mode in self.CLASSIFICATION_LOSS_MODES

    def _build_train_loader(self, train_data):
        collate = lambda items: stage1_collate_fn(items, max_len=self.args.seq_len)
        if not self._use_metric_loss():
            return DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=False,
                collate_fn=collate,
            )

        sampler = ClassBalancedBatchSampler(
            labels=train_data.y,
            batch_size=self.args.batch_size,
            samples_per_class=self.args.stage1_samples_per_class,
            drop_last=False,
            seed=self.args.seed,
        )
        return DataLoader(
            train_data,
            batch_sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=collate,
        )

    def _build_eval_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=lambda x: stage1_collate_fn(x, max_len=self.args.seq_len),
        )

    def _select_optimizer(self):
        weight_decay = float(getattr(self.args, "stage1_weight_decay", 0.0))
        if weight_decay < 0:
            raise ValueError("stage1_weight_decay must be non-negative")
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=weight_decay,
        )

    def _select_criterion(self):
        self.classification_criterion = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(self.args, "stage1_label_smoothing", 0.0))
        )

        if self._use_triplet_loss():
            self.triplet_miner = TripletMarginMiner(
                margin=self.args.stage1_triplet_margin,
                miner_type=self.args.stage1_triplet_type,
            )
            self.triplet_criterion = nn.TripletMarginLoss(
                margin=self.args.stage1_triplet_margin,
                p=2,
                reduction="mean",
            )
        else:
            self.triplet_miner = None
            self.triplet_criterion = None

        if self._use_multi_similarity_loss():
            if pml_miners is None or pml_losses is None:
                raise ImportError(
                    "Multi-Similarity training requires pytorch-metric-learning. "
                    "Install it with `pip install pytorch-metric-learning`."
                )
            self.ms_miner = pml_miners.MultiSimilarityMiner(
                epsilon=float(getattr(self.args, "stage1_ms_epsilon", 0.1))
            )
            self.ms_criterion = pml_losses.MultiSimilarityLoss(
                alpha=float(getattr(self.args, "stage1_ms_alpha", 2.0)),
                beta=float(getattr(self.args, "stage1_ms_beta", 50.0)),
                base=float(getattr(self.args, "stage1_ms_base", 0.5)),
            )
        else:
            self.ms_miner = None
            self.ms_criterion = None

    def _extract_features(self, batch_x):
        logits, fused_features = self.model(batch_x, return_features=True)
        embeddings = F.normalize(fused_features, p=2, dim=1)
        return logits, fused_features, embeddings

    def _compute_triplet_component(self, embeddings, labels):
        if not self._use_triplet_loss():
            return embeddings.sum() * 0.0, 0

        triplets = self.triplet_miner(embeddings, labels.long())
        if triplets is None:
            return embeddings.sum() * 0.0, 0

        anchor_idx, positive_idx, negative_idx = triplets
        loss = self.triplet_criterion(
            embeddings[anchor_idx],
            embeddings[positive_idx],
            embeddings[negative_idx],
        )
        return loss, int(anchor_idx.numel())

    def _compute_multi_similarity_component(self, embeddings, labels):
        if not self._use_multi_similarity_loss():
            return embeddings.sum() * 0.0, 0

        labels = labels.reshape(-1).long()
        hard_pairs = self.ms_miner(embeddings, labels)
        if hard_pairs is None:
            return embeddings.sum() * 0.0, 0

        # MultiSimilarityMiner returns (positive_anchor, positive, negative_anchor,
        # negative).  Count pairs rather than the four index tensors.
        if len(hard_pairs) == 4:
            pair_count = int(hard_pairs[0].numel() + hard_pairs[2].numel())
        else:
            pair_count = sum(int(pair_indices.numel()) for pair_indices in hard_pairs)
        if pair_count == 0:
            return embeddings.sum() * 0.0, 0

        loss = self.ms_criterion(
            embeddings,
            labels,
            indices_tuple=hard_pairs,
        )
        return loss, pair_count

    def _compute_metric_component(self, embeddings, labels):
        if self._use_triplet_loss():
            return self._compute_triplet_component(embeddings, labels)
        if self._use_multi_similarity_loss():
            return self._compute_multi_similarity_component(embeddings, labels)
        return embeddings.sum() * 0.0, 0

    def _compute_classification_component(self, logits, embeddings, labels):
        if not self._use_classification_loss():
            return embeddings.sum() * 0.0

        stage1_model = unwrap_model(self.model)
        if self.loss_mode in {"arcface", "arcface_triplet", "cosface", "cosface_triplet"}:
            if stage1_model.margin_head is None:
                raise RuntimeError(
                    f"stage1 margin head is missing for loss mode {self.loss_mode}"
                )
            margin_logits = stage1_model.margin_head(embeddings, labels.long())
            return self.classification_criterion(margin_logits, labels.long())

        return self.classification_criterion(logits, labels.long())

    def _compute_total_loss(self, logits, fused_features, embeddings, labels, clip_targets=None, resnet_targets=None):
        classification_loss = self._compute_classification_component(
            logits, embeddings, labels
        )
        metric_loss, mined_triplets = self._compute_metric_component(embeddings, labels)
        clip_loss = embeddings.sum() * 0.0
        resnet_loss = embeddings.sum() * 0.0
        if self.clip_enabled:
            if clip_targets is None:
                raise ValueError("CLIP alignment is enabled but this batch has no paired CLIP targets")
            stage1_model = unwrap_model(self.model)
            projected = F.normalize(stage1_model.clip_projection(fused_features).float(), p=2, dim=-1)
            clip_targets = F.normalize(clip_targets.to(self.device).float(), p=2, dim=-1)
            clip_loss = (1.0 - F.cosine_similarity(projected, clip_targets, dim=-1)).mean()
        if self.resnet_enabled:
            if resnet_targets is None:
                raise ValueError("ResNet50 alignment is enabled but this batch has no paired ResNet targets")
            stage1_model = unwrap_model(self.model)
            projected = F.normalize(stage1_model.resnet_projection(fused_features).float(), p=2, dim=-1)
            resnet_targets = F.normalize(resnet_targets.to(self.device).float(), p=2, dim=-1)
            resnet_loss = (1.0 - F.cosine_similarity(projected, resnet_targets, dim=-1)).mean()

        total_loss = logits.sum() * 0.0
        if self._use_classification_loss():
            total_loss = total_loss + self.args.stage1_ce_weight * classification_loss
        if self._use_metric_loss():
            total_loss = total_loss + self.args.stage1_triplet_weight * metric_loss
        if self.clip_enabled:
            total_loss = total_loss + self.args.stage1_clip_weight * clip_loss
        if self.resnet_enabled:
            total_loss = total_loss + self.args.stage1_resnet_weight * resnet_loss

        return (
            total_loss,
            float(classification_loss.detach().item()),
            float(metric_loss.detach().item()),
            float(clip_loss.detach().item()),
            float(resnet_loss.detach().item()),
            mined_triplets,
        )

    def _evaluate_loader(self, data_loader, desc=None):
        total_loss = []
        classification_losses = []
        metric_losses = []
        clip_losses = []
        resnet_losses = []
        feature_rows = []
        label_rows = []
        mined_triplets = 0

        self.model.eval()
        with torch.no_grad():
            iterator = (
                tqdm(
                    data_loader,
                    desc=desc,
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                )
                if desc
                else data_loader
            )
            for batch in iterator:
                batch_x, label, padding_mask = batch[:3]
                clip_targets = batch[3] if len(batch) > 3 else None
                resnet_targets = batch[4] if len(batch) > 4 else None
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                logits, fused_features, embeddings = self._extract_features(batch_x)
                loss, classification_loss, metric_loss, clip_loss, resnet_loss, batch_triplets = self._compute_total_loss(
                    logits, fused_features, embeddings, label, clip_targets, resnet_targets
                )
                total_loss.append(loss.item())
                classification_losses.append(classification_loss)
                metric_losses.append(metric_loss)
                clip_losses.append(clip_loss)
                resnet_losses.append(resnet_loss)
                mined_triplets += batch_triplets
                feature_rows.append(embeddings.cpu().numpy())
                label_rows.append(label.detach().cpu().numpy())
                if desc:
                    iterator.set_postfix(
                        loss=f"{loss.item():.4f}",
                        refresh=False,
                    )

        total_loss = float(np.average(total_loss))
        classification_loss = float(np.average(classification_losses))
        metric_loss = float(np.average(metric_losses))
        clip_loss = float(np.average(clip_losses))
        resnet_loss = float(np.average(resnet_losses))
        embeddings = np.concatenate(feature_rows, axis=0)
        labels = np.concatenate(label_rows, axis=0)
        kmeans_score = compute_kmeans_score(
            embeddings,
            labels,
            n_clusters=self._feature_kmeans_clusters(),
            random_state=self.args.seed,
        )
        self.model.train()
        return total_loss, classification_loss, metric_loss, clip_loss, resnet_loss, kmeans_score, mined_triplets

    def _maybe_resume_from_checkpoint(self):
        checkpoint_path = getattr(self.args, "resume_ckpt", None)
        if checkpoint_path is None:
            return None

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = load_checkpoint_object(checkpoint_path, map_location=self.device)
        try:
            load_model_state(self.model, checkpoint)
            print(f"Loaded checkpoint for stage1 training: {checkpoint_path}")
        except RuntimeError:
            state_dict = extract_model_state(checkpoint)
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
            incompatible = unwrap_model(self.model).load_state_dict(state_dict, strict=False)
            allowed_prefixes = ("clip_projection.", "resnet_projection.")
            invalid_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
            invalid_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(allowed_prefixes)]
            if invalid_missing or invalid_unexpected:
                raise RuntimeError(
                    "Resume checkpoint is incompatible with the current Stage-1 model; "
                    f"missing={invalid_missing}, unexpected={invalid_unexpected}"
                )
            print(
                f"Loaded compatible Stage-1 weights from {checkpoint_path}; "
                "new or disabled image projection heads were initialized according to current parameters."
            )
        return checkpoint

    def train(self, setting):
        raw_train_data, _ = self._get_data(flag="TRAIN")
        raw_vali_data, _ = self._get_data(flag="VAL")
        test_data, _ = self._get_data(flag="TEST")

        raw_train_data = self._with_image_targets(raw_train_data)
        raw_vali_data = self._with_image_targets(raw_vali_data)
        test_data = self._with_image_targets(test_data)
        self._release_image_teachers()

        if self.use_validation:
            train_data = raw_train_data
            vali_data = raw_vali_data
            selection_name = "Validation"
            selection_split = "val"
        else:
            # Explicitly fold the original validation samples into training.
            # The test split is then the only held-out split and is used for
            # checkpoint selection as requested.
            train_data = self._merge_datasets(raw_train_data, raw_vali_data)
            vali_data = None
            selection_name = "Test"
            selection_split = "test"

        train_loader = self._build_train_loader(train_data)
        train_eval_loader = self._build_eval_loader(train_data)
        selection_loader = self._build_eval_loader(
            raw_vali_data if self.use_validation else test_data
        )

        print(self._dataset_summary(train_data, "TRAIN"))
        if self.use_validation:
            print(self._dataset_summary(raw_vali_data, "VAL"))
        else:
            print("VAL: disabled (original validation samples merged into TRAIN)")
        print(self._dataset_summary(test_data, "TEST"))
        self._prepare_metric_logging(setting, train_data, vali_data, test_data)

        checkpoint_dir = self._checkpoint_dir(setting)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        self._select_criterion()
        self._maybe_resume_from_checkpoint()

        configured_selection = str(getattr(self.args, "stage1_selection_metric", "auto")).lower()
        if configured_selection == "auto":
            selection_metric_name = "loss" if self.image_alignment_enabled else "kmeans_score"
        else:
            selection_metric_name = configured_selection
        maximize_selection_metric = selection_metric_name == "kmeans_score"
        best_selection_metric = -np.inf if maximize_selection_metric else np.inf
        early_stop_counter = 0

        epoch_iterator = tqdm(
            range(self.args.train_epochs),
            desc="Stage-1 training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            step_total_losses = []
            step_classification_losses = []
            step_metric_losses = []
            step_clip_losses = []
            step_resnet_losses = []
            step_mined_triplets = []

            self.model.train()
            epoch_time = time.time()
            batch_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_iterator:
                batch_x, label, padding_mask = batch[:3]
                clip_targets = batch[3] if len(batch) > 3 else None
                resnet_targets = batch[4] if len(batch) > 4 else None
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                logits, fused_features, embeddings = self._extract_features(batch_x)
                loss, classification_loss, metric_loss, clip_loss, resnet_loss, batch_triplets = self._compute_total_loss(
                    logits, fused_features, embeddings, label, clip_targets, resnet_targets
                )

                step_total_losses.append(loss.item())
                step_classification_losses.append(classification_loss)
                step_metric_losses.append(metric_loss)
                step_clip_losses.append(clip_loss)
                step_resnet_losses.append(resnet_loss)
                step_mined_triplets.append(batch_triplets)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                batch_iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    refresh=False,
                )

            epoch_time_cost = time.time() - epoch_time
            tqdm.write(
                f"Epoch: {epoch + 1} cost time: {epoch_time_cost}"
            )
            train_step_loss = float(np.average(step_total_losses))
            train_step_cls_loss = float(np.average(step_classification_losses))
            train_step_metric_loss = float(np.average(step_metric_losses))
            train_step_clip_loss = float(np.average(step_clip_losses))
            train_step_resnet_loss = float(np.average(step_resnet_losses))
            train_step_triplets = int(np.sum(step_mined_triplets))

            (
                train_eval_loss,
                train_eval_cls_loss,
                train_eval_metric_loss,
                train_eval_clip_loss,
                train_eval_resnet_loss,
                train_kmeans,
                train_eval_triplets,
            ) = self._evaluate_loader(
                train_eval_loader,
                desc=f"Epoch {epoch + 1} train eval",
            )
            (
                selection_loss,
                selection_cls_loss,
                selection_metric_loss,
                selection_clip_loss,
                selection_resnet_loss,
                selection_kmeans,
                selection_triplets,
            ) = self._evaluate_loader(
                selection_loader,
                desc=f"Epoch {epoch + 1} {selection_name.lower()} eval",
            )
            current_lr = model_optim.param_groups[0]["lr"]

            self._append_metric_row(
                epoch=epoch + 1,
                split="train",
                loss=train_eval_loss,
                classification_loss=train_eval_cls_loss,
                metric_loss=train_eval_metric_loss,
                clip_loss=train_eval_clip_loss,
                resnet_loss=train_eval_resnet_loss,
                kmeans_score=train_kmeans,
                mined_triplets=train_eval_triplets,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )
            self._append_metric_row(
                epoch=epoch + 1,
                split=selection_split,
                loss=selection_loss,
                classification_loss=selection_cls_loss,
                metric_loss=selection_metric_loss,
                clip_loss=selection_clip_loss,
                resnet_loss=selection_resnet_loss,
                kmeans_score=selection_kmeans,
                mined_triplets=selection_triplets,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )

            tqdm.write(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | "
                f"Train Step Loss: {train_step_loss:.5f}, "
                f"Cls: {train_step_cls_loss:.5f}, Metric: {train_step_metric_loss:.5f}, "
                f"CLIP: {train_step_clip_loss:.5f}, ResNet: {train_step_resnet_loss:.5f}\n"
                f"Train feature results --- Loss: {train_eval_loss:.5f}, Cls: {train_eval_cls_loss:.5f}, "
                f"Metric: {train_eval_metric_loss:.5f}, CLIP: {train_eval_clip_loss:.5f}, ResNet: {train_eval_resnet_loss:.5f}, "
                f"KMeans: {train_kmeans:.5f}, "
                f"MinedTriplets(step/eval): {train_step_triplets}/{train_eval_triplets}\n"
                f"{selection_name} feature results --- Loss: {selection_loss:.5f}, "
                f"Cls: {selection_cls_loss:.5f}, Metric: {selection_metric_loss:.5f}, "
                f"CLIP: {selection_clip_loss:.5f}, ResNet: {selection_resnet_loss:.5f}, "
                f"KMeans: {selection_kmeans:.5f}, MinedTriplets: {selection_triplets}"
            )
            weighted_image_loss = (
                float(getattr(self.args, "stage1_clip_weight", 0.0)) * selection_clip_loss
                + float(getattr(self.args, "stage1_resnet_weight", 0.0)) * selection_resnet_loss
            )
            selection_values = {
                "loss": selection_loss,
                "kmeans_score": selection_kmeans,
                "image_loss": weighted_image_loss,
            }
            selection_value = selection_values[selection_metric_name]
            epoch_iterator.set_postfix(
                train_loss=f"{train_eval_loss:.4f}",
                selection=f"{selection_value:.4f}",
                refresh=False,
            )
            if maximize_selection_metric:
                improved = selection_value > best_selection_metric + 1e-6
            else:
                improved = selection_value < best_selection_metric - 1e-6

            if improved:
                best_selection_metric = selection_value
                early_stop_counter = 0
                stage1_model = unwrap_model(self.model)
                checkpoint_payload = build_checkpoint_payload(
                    self.model,
                    optimizer=model_optim,
                    epoch=epoch,
                    best_metric=best_selection_metric,
                    args_dict=vars(deepcopy(self.args)),
                    extra={
                        "stage": "stage1_feature_training",
                        "selection_split": selection_split,
                        "selection_metric": selection_metric_name,
                        "selection_metric_value": float(best_selection_metric),
                        "backbone_state_dict": stage1_model.backbone.state_dict(),
                    },
                )
                torch.save(checkpoint_payload, checkpoint_dir / "checkpoint.pth")
                tqdm.write(
                    f"{selection_name} {selection_metric_name} improved to {best_selection_metric:.5f}. "
                    "Saving best feature checkpoint ..."
                )
            else:
                early_stop_counter += 1
                print(
                    f"EarlyStopping counter: {early_stop_counter} out of {self.args.patience}"
                )
                if early_stop_counter >= self.args.patience:
                    print("Early stopping")
                    break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = checkpoint_dir / "checkpoint.pth"
        if best_model_path.exists():
            load_model_state(self.model, best_model_path, map_location=self.device)
        return self.model

    def test(self, setting, test=0):
        test_data, _ = self._get_data(flag="TEST")
        test_data = self._with_image_targets(test_data)
        if self.use_validation:
            vali_data, _ = self._get_data(flag="VAL")
            vali_data = self._with_image_targets(vali_data)
            vali_loader = self._build_eval_loader(vali_data)
        else:
            vali_loader = None
        self._release_image_teachers()
        test_loader = self._build_eval_loader(test_data)
        checkpoint_dir = self._checkpoint_dir(setting)

        best_model_path = checkpoint_dir / "checkpoint.pth"
        if best_model_path.exists():
            load_model_state(self.model, best_model_path, map_location=self.device)

        self._select_criterion()
        (
            test_loss,
            test_cls_loss,
            test_metric_loss,
            test_clip_loss,
            test_resnet_loss,
            test_kmeans,
            test_triplets,
        ) = self._evaluate_loader(test_loader)

        if self.use_validation:
            (
                val_loss,
                val_cls_loss,
                val_metric_loss,
                val_clip_loss,
                val_resnet_loss,
                val_kmeans,
                val_triplets,
            ) = self._evaluate_loader(vali_loader)
            self._append_metric_row(
                epoch="final",
                split="val_final",
                loss=val_loss,
                classification_loss=val_cls_loss,
                metric_loss=val_metric_loss,
                clip_loss=val_clip_loss,
                resnet_loss=val_resnet_loss,
                kmeans_score=val_kmeans,
                mined_triplets=val_triplets,
            )
        else:
            val_loss = val_cls_loss = val_metric_loss = val_clip_loss = val_resnet_loss = val_kmeans = val_triplets = None

        self._append_metric_row(
            epoch="final",
            split="test_final",
            loss=test_loss,
            classification_loss=test_cls_loss,
            metric_loss=test_metric_loss,
            clip_loss=test_clip_loss,
            resnet_loss=test_resnet_loss,
            kmeans_score=test_kmeans,
            mined_triplets=test_triplets,
        )

        if self.use_validation:
            print(
                f"Final validation feature results --- Loss: {val_loss:.5f}, Cls: {val_cls_loss:.5f}, "
                f"Metric: {val_metric_loss:.5f}, CLIP: {val_clip_loss:.5f}, ResNet: {val_resnet_loss:.5f}, KMeans: {val_kmeans:.5f}, MinedTriplets: {val_triplets}\n"
                f"Final test feature results --- Loss: {test_loss:.5f}, Cls: {test_cls_loss:.5f}, "
                f"Metric: {test_metric_loss:.5f}, CLIP: {test_clip_loss:.5f}, ResNet: {test_resnet_loss:.5f}, KMeans: {test_kmeans:.5f}, MinedTriplets: {test_triplets}"
            )
        else:
            print(
                f"Final test feature results (validation disabled) --- Loss: {test_loss:.5f}, "
                f"Cls: {test_cls_loss:.5f}, Metric: {test_metric_loss:.5f}, CLIP: {test_clip_loss:.5f}, ResNet: {test_resnet_loss:.5f}, "
                f"KMeans: {test_kmeans:.5f}, MinedTriplets: {test_triplets}"
            )

        if not getattr(self.args, "keep_checkpoint", True):
            self.del_weight(str(checkpoint_dir))

        return {
            "ValKMeans": np.nan if val_kmeans is None else val_kmeans,
            "TestKMeans": test_kmeans,
        }

    def del_weight(self, path):
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Model weights deleted....")
