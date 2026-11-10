"""Align DA4FE EEG features with paired frozen CLIP image embeddings."""

import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_provider.generation_dataset import (
    EEGImageNetHuggingFaceGenerationDataset,
    generation_collate,
)
from exp.exp_stage1_feature import Stage1FeatureModel
from models import DA4FE
from models.eeg_clip_alignment import EEGImageProjection
from utils.checkpointing import load_checkpoint_object, load_model_state


def _as_namespace(payload):
    return SimpleNamespace(**(payload.get("args", payload) if isinstance(payload, dict) else {}))


def _first_existing(paths):
    for path in paths:
        if path is not None and Path(path).is_file():
            return Path(path)
    return None


class Exp_Stage3A_ImageAlignment:
    """Train an EEG projection against frozen CLIP embeddings of paired images."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.stage1_args, self.stage1_checkpoint_path = self._load_stage1_metadata()
        self._apply_stage1_overrides()
        dataset_args = self._dataset_args()
        self.train_data = EEGImageNetHuggingFaceGenerationDataset(dataset_args, "TRAIN")
        self.val_data = EEGImageNetHuggingFaceGenerationDataset(dataset_args, "VAL")
        self.test_data = EEGImageNetHuggingFaceGenerationDataset(dataset_args, "TEST")
        self.stage1_model = self._build_stage1_model().to(self.device)
        self.feature_dim = int(self.stage1_model.feature_dim)
        self.eeg_image_projection, self.clip_model = self._build_alignment_models()
        self._load_resume_checkpoint()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch", "split", "loss", "alignment_loss", "classification_loss",
                    "recall_at_1", "recall_at_5", "paired_cosine", "epoch_time_sec",
                ],
            ).writeheader()

    def _load_stage1_metadata(self):
        run_dir = Path(self.args.stage1_run_dir).expanduser() if self.args.stage1_run_dir else None
        if run_dir is not None and run_dir.is_file():
            run_dir = run_dir.parent
        run_params_path = _first_existing([
            self.args.stage1_run_params,
            run_dir / "run_params.json" if run_dir else None,
        ])
        if run_params_path is None:
            raise FileNotFoundError("Could not find run_params.json; pass --stage1_run_dir or --stage1_run_params")
        with run_params_path.open("r", encoding="utf-8") as handle:
            stage1_args = _as_namespace(json.load(handle))

        split_summary_path = _first_existing([
            self.args.stage1_split_summary,
            run_dir / "split_summary.json" if run_dir else None,
        ])
        self.stage1_split_summary = None
        if split_summary_path is not None:
            with split_summary_path.open("r", encoding="utf-8") as handle:
                self.stage1_split_summary = json.load(handle)

        checkpoint_path = _first_existing([
            self.args.stage1_checkpoint,
            run_dir / "checkpoint.pth" if run_dir else None,
        ])
        if checkpoint_path is None:
            raise FileNotFoundError("Could not find checkpoint.pth; pass --stage1_checkpoint")
        self.stage1_run_params_path = run_params_path
        self.stage1_split_summary_path = split_summary_path
        return stage1_args, checkpoint_path

    def _apply_stage1_overrides(self):
        names = (
            "eeg_hf_dataset_id", "eeg_hf_cache_dir", "eeg_hf_split_mode",
            "eeg_hf_split_seed", "eeg_hf_train_ratio", "eeg_hf_val_ratio",
            "eeg_hf_test_ratio", "eeg_filter_enabled", "eeg_filter_low_hz",
            "eeg_filter_high_hz",
        )
        for name in names:
            value = getattr(self.args, name, None)
            if value is not None:
                setattr(self.stage1_args, name, value)
        self.stage1_args.requested_seq_len = getattr(
            self.stage1_args, "requested_seq_len", getattr(self.stage1_args, "seq_len", 512)
        )
        self.stage1_args.seq_len = int(getattr(self.stage1_args, "seq_len", self.stage1_args.requested_seq_len))
        self.stage1_args.eeg_adaptive_seq_len = getattr(self.stage1_args, "eeg_adaptive_seq_len", True)
        self.stage1_args.eeg_normalize = getattr(self.stage1_args, "eeg_normalize", True)
        self.stage1_args.seed = int(getattr(self.stage1_args, "seed", self.args.seed))
        for name, default in (("eeg_filter_enabled", False), ("eeg_filter_low_hz", 0.0), ("eeg_filter_high_hz", 45.0)):
            value = getattr(self.stage1_args, name, default)
            if value is None:
                value = default
            setattr(self.stage1_args, name, value)
            setattr(self.args, name, value)

    def _dataset_args(self):
        dataset_args = SimpleNamespace(**vars(self.stage1_args))
        dataset_args.generation_image_size = self.args.stage3a_image_size
        dataset_args.generation_random_crop = False
        dataset_args.eeg_hf_split_mode = getattr(self.stage1_args, "eeg_hf_split_mode", "original")
        dataset_args.eeg_hf_split_seed = getattr(self.stage1_args, "eeg_hf_split_seed", self.args.seed)
        dataset_args.num_workers = self.args.num_workers
        return dataset_args

    def _build_stage1_model(self):
        stage1_args = self.stage1_args
        stage1_args.model = getattr(stage1_args, "model", "DA4FE")
        if stage1_args.model != "DA4FE":
            raise ValueError(f"Stage-3A currently expects a DA4FE checkpoint, got {stage1_args.model!r}")
        stage1_args.num_class = int(getattr(stage1_args, "num_class", 0) or self.train_data.num_class)
        stage1_args.enc_in = int(getattr(stage1_args, "enc_in", self.train_data.feature_dim))
        stage1_args.use_multi_gpu = False
        stage1_args.stage1_l2_normalize = getattr(stage1_args, "stage1_l2_normalize", True)
        backbone = DA4FE.Model(stage1_args).float()
        feature_dim = int(getattr(
            stage1_args,
            "stage1_backbone_feature_dim",
            getattr(stage1_args, "da4fe_fusion_out_dim", 512),
        ))
        model = Stage1FeatureModel(
            backbone=backbone,
            feature_dim=feature_dim,
            num_class=stage1_args.num_class,
            args=stage1_args,
        ).float()
        checkpoint = load_checkpoint_object(self.stage1_checkpoint_path, map_location="cpu")
        try:
            load_model_state(model, checkpoint)
        except RuntimeError:
            if isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
                state = {key.removeprefix("module."): value for key, value in checkpoint["backbone_state_dict"].items()}
                model.backbone.load_state_dict(state)
            else:
                raise
        for parameter in model.backbone.parameters():
            parameter.requires_grad = bool(self.args.stage3a_train_da4fe)
        model.eval()
        return model

    def _resolve_clip_path(self):
        explicit = self.args.clip_model_path
        path = Path(explicit).expanduser() if explicit else Path(self.args.pretrained_root).expanduser() / "clip-vit-large-patch14"
        if path.exists():
            return str(path)
        if self.args.allow_hf_download:
            return str(explicit) if explicit and "/" in str(explicit).replace("\\", "/") else "openai/clip-vit-large-patch14"
        raise FileNotFoundError(f"CLIP model path does not exist: {path}; enable --allow_hf_download or set --clip_model_path")

    def _build_alignment_models(self):
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "Stage-3A requires transformers>=4.40.0; install the pinned project version "
                "in the active environment with: python -m pip install transformers==4.44.2"
            ) from exc
        clip_model = CLIPVisionModelWithProjection.from_pretrained(
            self._resolve_clip_path(),
            cache_dir=self.args.hf_cache_dir,
            local_files_only=not self.args.allow_hf_download,
        ).to(self.device).eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False
        clip_dim = getattr(clip_model.config, "projection_dim", None)
        if clip_dim is None:
            clip_dim = clip_model.visual_projection.out_features
        clip_dim = int(clip_dim)
        projection = EEGImageProjection(
            input_dim=self.feature_dim,
            output_dim=clip_dim,
            hidden_dim=self.args.stage3a_projection_hidden_dim,
            dropout=self.args.stage3a_projection_dropout,
        ).to(self.device)
        self.clip_feature_dim = clip_dim
        self.clip_target = "projected_image_embeds"
        self.projection_config = {
            "input_dim": self.feature_dim,
            "output_dim": clip_dim,
            "hidden_dim": self.args.stage3a_projection_hidden_dim,
            "dropout": self.args.stage3a_projection_dropout,
        }
        parameter_groups = [{
            "params": list(projection.parameters()),
            "lr": self.args.stage3a_projection_lr,
        }]
        if self.args.stage3a_train_da4fe:
            parameter_groups.append({
                "params": [parameter for parameter in self.stage1_model.backbone.parameters() if parameter.requires_grad],
                "lr": self.args.stage3a_da4fe_lr,
            })
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.args.stage3a_weight_decay,
        )
        return projection, clip_model

    def _load_resume_checkpoint(self):
        resume_path = getattr(self.args, "stage3a_resume_checkpoint", None)
        if not resume_path:
            return
        resume_path = Path(resume_path).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Stage-3A resume checkpoint does not exist: {resume_path}")
        payload = load_checkpoint_object(resume_path, map_location="cpu")
        if not isinstance(payload, dict) or "stage1_model" not in payload or "eeg_image_projection" not in payload:
            raise ValueError(f"Not a compatible Stage-3A checkpoint: {resume_path}")
        if payload.get("clip_target") != self.clip_target:
            raise ValueError(
                "Stage-3A resume checkpoint uses a different CLIP target representation; "
                "checkpoints trained against pooled vision features cannot resume against projected image embeddings"
            )
        if int(payload.get("feature_dim", -1)) != self.feature_dim or int(payload.get("clip_feature_dim", -1)) != self.clip_feature_dim:
            raise ValueError("Stage-3A resume checkpoint dimensions do not match the current DA4FE/CLIP models")
        self.stage1_model.load_state_dict(payload["stage1_model"])
        self.eeg_image_projection.load_state_dict(payload["eeg_image_projection"])
        print(f"Resumed Stage-3A encoder and projection weights from: {resume_path}")

    def _loader(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.args.stage3a_batch_size,
            shuffle=train,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=lambda batch: generation_collate(batch, max_len=self.stage1_args.seq_len),
        )

    def _clip_values(self, images):
        images = (images + 1.0) / 2.0
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        return (images - mean) / std

    def _encode_clip_images(self, images):
        output = self.clip_model(pixel_values=self._clip_values(images))
        return output.image_embeds.float()

    @staticmethod
    def _paired_cosine_loss(eeg_embeddings, image_embeddings):
        """Minimize cosine distance between each EEG and its paired CLIP image embedding."""
        return (1.0 - F.cosine_similarity(eeg_embeddings, image_embeddings, dim=-1)).mean()

    def _train_batch(self, batch):
        eeg = batch["eeg"].to(self.device, non_blocking=True).float()
        images = batch["image"].to(self.device, non_blocking=True).float()
        labels = batch["label"].to(self.device, non_blocking=True).long()
        with torch.set_grad_enabled(self.args.stage3a_train_da4fe):
            logits, features = self.stage1_model(eeg, return_features=True)
        projected = self.eeg_image_projection(features)
        with torch.no_grad():
            image_features = self._encode_clip_images(images)
        eeg_embeddings = F.normalize(projected.float(), dim=-1)
        image_embeddings = F.normalize(image_features.float(), dim=-1)
        alignment_loss = self._paired_cosine_loss(eeg_embeddings, image_embeddings)
        classification_loss = torch.zeros((), device=self.device)
        if self.args.stage3a_train_da4fe and self.args.stage3a_classification_weight > 0:
            classification_loss = F.cross_entropy(logits.float(), labels)
        total = alignment_loss + self.args.stage3a_classification_weight * classification_loss
        return total, alignment_loss.detach(), classification_loss.detach()

    def _append_metric(self, epoch, split, metrics, epoch_time=""):
        row = {
            "epoch": epoch,
            "split": split,
            "loss": metrics["loss"],
            "alignment_loss": metrics["alignment_loss"],
            "classification_loss": metrics["classification_loss"],
            "recall_at_1": metrics["recall_at_1"],
            "recall_at_5": metrics["recall_at_5"],
            "paired_cosine": metrics["paired_cosine"],
            "epoch_time_sec": epoch_time,
        }
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=row.keys()).writerow(row)

    def _evaluate(self, dataset, split):
        self.stage1_model.eval()
        self.eeg_image_projection.eval()
        self.clip_model.eval()
        eeg_values, image_values, image_ids = [], [], []
        alignment_total = classification_total = total_count = 0
        with torch.no_grad():
            for batch in tqdm(self._loader(dataset), desc=f"Evaluate 3A {split}", leave=False):
                eeg = batch["eeg"].to(self.device, non_blocking=True).float()
                images = batch["image"].to(self.device, non_blocking=True).float()
                logits, features = self.stage1_model(eeg, return_features=True)
                projected = F.normalize(self.eeg_image_projection(features).float(), dim=-1)
                image_features = F.normalize(self._encode_clip_images(images).float(), dim=-1)
                count = len(batch["image_id"])
                alignment_total += float(self._paired_cosine_loss(projected, image_features)) * count
                if self.args.stage3a_train_da4fe and self.args.stage3a_classification_weight > 0:
                    classification_total += float(F.cross_entropy(logits.float(), batch["label"].to(self.device).long())) * count
                total_count += count
                eeg_values.append(projected.cpu())
                image_values.append(image_features.cpu())
                image_ids.extend(batch["image_id"])

        eeg_values = torch.cat(eeg_values).to(self.device)
        image_values = torch.cat(image_values).to(self.device)
        ids = np.asarray(image_ids, dtype=object)
        hits1, hits5 = [], []
        gallery_size = len(image_ids)
        top_k = min(5, gallery_size)
        for start in range(0, gallery_size, 128):
            stop = min(start + 128, gallery_size)
            scores = eeg_values[start:stop] @ image_values.T
            indices = scores.topk(top_k, dim=1).indices.cpu().numpy()
            retrieved_ids = ids[indices]
            matches = retrieved_ids == ids[start:stop, None]
            hits1.extend(matches[:, :1].any(axis=1).astype(np.float32).tolist())
            hits5.extend(matches.any(axis=1).astype(np.float32).tolist())

        paired_cosine = F.cosine_similarity(eeg_values, image_values, dim=-1).mean().item()
        alignment_loss = alignment_total / max(total_count, 1)
        classification_loss = classification_total / max(total_count, 1)
        result = {
            "loss": alignment_loss + (
                self.args.stage3a_classification_weight * classification_loss
                if self.args.stage3a_train_da4fe else 0.0
            ),
            "alignment_loss": alignment_loss,
            "classification_loss": classification_loss,
            "recall_at_1": float(np.mean(hits1)),
            "recall_at_5": float(np.mean(hits5)),
            "paired_cosine": float(paired_cosine),
        }
        print(
            f"3A {split}: align_loss={result['alignment_loss']:.5f}, "
            f"R@1={result['recall_at_1']:.4f}, R@5={result['recall_at_5']:.4f}, "
            f"paired_cos={result['paired_cosine']:.4f}"
        )
        return result

    def _save_checkpoint(self, epoch, metrics):
        payload = {
            "stage3a_version": 2,
            "clip_target": self.clip_target,
            "epoch": int(epoch),
            "metrics": metrics,
            "stage1_model": self.stage1_model.state_dict(),
            "eeg_image_projection": self.eeg_image_projection.state_dict(),
            "projection_config": self.projection_config,
            "feature_dim": self.feature_dim,
            "clip_feature_dim": self.clip_feature_dim,
            "stage1_args": vars(self.stage1_args),
            "stage1_checkpoint": str(self.stage1_checkpoint_path.resolve()),
            "stage1_run_params": str(self.stage1_run_params_path.resolve()),
            "split_summary": self.stage1_split_summary,
            "optimizer": self.optimizer.state_dict(),
        }
        path = self.output_dir / "checkpoint.pth"
        torch.save(payload, path)
        return path

    def _load_best_checkpoint(self):
        path = self.output_dir / "checkpoint.pth"
        if not path.is_file():
            return
        payload = torch.load(path, map_location=self.device)
        self.stage1_model.load_state_dict(payload["stage1_model"])
        self.eeg_image_projection.load_state_dict(payload["eeg_image_projection"])

    def train(self):
        best_loss = math.inf
        best_recall = -math.inf
        history = []
        progress = tqdm(range(1, self.args.stage3a_epochs + 1), desc="Stage-3A total", unit="epoch")
        for epoch in progress:
            self.stage1_model.train(self.args.stage3a_train_da4fe)
            self.eeg_image_projection.train()
            totals = np.zeros(3, dtype=np.float64)
            count = 0
            start = time.time()
            for batch in tqdm(self._loader(self.train_data, train=True), desc=f"Epoch {epoch}/{self.args.stage3a_epochs}", leave=False):
                self.optimizer.zero_grad(set_to_none=True)
                total, alignment, classification = self._train_batch(batch)
                total.backward()
                trainable = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
                torch.nn.utils.clip_grad_norm_(trainable, self.args.stage3a_grad_clip)
                self.optimizer.step()
                size = len(batch["image_id"])
                totals += np.asarray([float(total), float(alignment), float(classification)]) * size
                count += size
            train_metrics = {
                "loss": totals[0] / max(count, 1),
                "alignment_loss": totals[1] / max(count, 1),
                "classification_loss": totals[2] / max(count, 1),
                "recall_at_1": "",
                "recall_at_5": "",
                "paired_cosine": "",
            }
            val_metrics = self._evaluate(self.val_data, "val")
            epoch_time = time.time() - start
            self._append_metric(epoch, "train", train_metrics, epoch_time)
            self._append_metric(epoch, "val", val_metrics, epoch_time)
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "epoch_time_sec": epoch_time})
            is_better = (
                val_metrics["loss"] < best_loss
                or (math.isclose(val_metrics["loss"], best_loss) and val_metrics["recall_at_1"] > best_recall)
            )
            if is_better:
                best_loss = val_metrics["loss"]
                best_recall = val_metrics["recall_at_1"]
                self._save_checkpoint(epoch, history[-1])
            progress.set_postfix(train=f"{train_metrics['loss']:.4f}", val_r1=f"{val_metrics['recall_at_1']:.3f}")
            print(
                f"Epoch {epoch}: train_align_loss={train_metrics['alignment_loss']:.5f}, "
                f"val_R@1={val_metrics['recall_at_1']:.4f}"
            )

        self._load_best_checkpoint()
        test_metrics = self._evaluate(self.test_data, "test")
        self._append_metric("best", "test", test_metrics)
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        return history, test_metrics
