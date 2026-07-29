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
from torch.utils.data import BatchSampler, DataLoader

from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from exp.exp_basic import Exp_Basic
from utils.checkpointing import (
    build_checkpoint_payload,
    load_checkpoint_object,
    load_model_state,
    unwrap_model,
)
from utils.stage_metrics import compute_kmeans_score
from utils.tools import adjust_learning_rate

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
    def __init__(self, backbone, feature_dim, num_class, args):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.loss_mode = getattr(args, "stage1_loss", "triplet").lower()
        self.margin_head = None

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
                "stage1_samples_per_class must be >= 2 because triplet loss "
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


class Exp_Stage1_Feature(Exp_Basic):
    METRIC_FIELDNAMES = [
        "epoch",
        "split",
        "loss",
        "classification_loss",
        "metric_loss",
        "kmeans_score",
        "mined_triplets",
        "learning_rate",
        "epoch_time_sec",
        "steps",
    ]

    TRIPLET_LOSS_MODES = {"triplet", "ce_triplet", "arcface_triplet", "cosface_triplet"}
    CLASSIFICATION_LOSS_MODES = {"ce", "ce_triplet", "arcface", "arcface_triplet", "cosface", "cosface_triplet"}

    def __init__(self, args):
        self.loss_mode = getattr(args, "stage1_loss", "triplet").lower()
        self.triplet_miner = None
        self.triplet_criterion = None
        self.classification_criterion = None
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

    def _dataset_summary(self, dataset, split_name):
        seq_len = self._dataset_seq_len(dataset)
        feature_dim = self._dataset_feature_dim(dataset)
        num_class = self._dataset_num_class(dataset)
        return (
            f"{split_name}: samples={len(dataset)}, seq_len={seq_len}, "
            f"feat_dim={feature_dim}, num_class={num_class}"
        )

    def _dataset_summary_payload(self, dataset, split_name):
        return {
            "split": split_name,
            "samples": len(dataset),
            "seq_len": self._dataset_seq_len(dataset),
            "feat_dim": self._dataset_feature_dim(dataset),
            "num_class": self._dataset_num_class(dataset),
        }

    def _checkpoint_dir(self, setting):
        return Path("./checkpoints") / self.args.model / "stage1" / setting

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

        model = Stage1FeatureModel(
            backbone=backbone,
            feature_dim=feature_dim,
            num_class=self.args.num_class,
            args=self.args,
        ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        return data_provider(self.args, flag)

    def _use_triplet_loss(self):
        return self.loss_mode in self.TRIPLET_LOSS_MODES

    def _use_classification_loss(self):
        return self.loss_mode in self.CLASSIFICATION_LOSS_MODES

    def _build_train_loader(self, train_data):
        if not self._use_triplet_loss():
            return DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=False,
                collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
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
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
        )

    def _build_eval_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
        )

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

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

    def _compute_total_loss(self, logits, embeddings, labels):
        classification_loss = self._compute_classification_component(
            logits, embeddings, labels
        )
        metric_loss, mined_triplets = self._compute_triplet_component(embeddings, labels)

        total_loss = logits.sum() * 0.0
        if self._use_classification_loss():
            total_loss = total_loss + self.args.stage1_ce_weight * classification_loss
        if self._use_triplet_loss():
            total_loss = total_loss + self.args.stage1_triplet_weight * metric_loss

        return (
            total_loss,
            float(classification_loss.detach().item()),
            float(metric_loss.detach().item()),
            mined_triplets,
        )

    def _evaluate_loader(self, data_loader):
        total_loss = []
        classification_losses = []
        metric_losses = []
        feature_rows = []
        label_rows = []
        mined_triplets = 0

        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                logits, fused_features, embeddings = self._extract_features(batch_x)
                loss, classification_loss, metric_loss, batch_triplets = self._compute_total_loss(
                    logits, embeddings, label
                )
                total_loss.append(loss.item())
                classification_losses.append(classification_loss)
                metric_losses.append(metric_loss)
                mined_triplets += batch_triplets
                feature_rows.append(embeddings.cpu().numpy())
                label_rows.append(label.detach().cpu().numpy())

        total_loss = float(np.average(total_loss))
        classification_loss = float(np.average(classification_losses))
        metric_loss = float(np.average(metric_losses))
        embeddings = np.concatenate(feature_rows, axis=0)
        labels = np.concatenate(label_rows, axis=0)
        kmeans_score = compute_kmeans_score(
            embeddings,
            labels,
            n_clusters=self._feature_kmeans_clusters(),
            random_state=self.args.seed,
        )
        self.model.train()
        return total_loss, classification_loss, metric_loss, kmeans_score, mined_triplets

    def _maybe_resume_from_checkpoint(self):
        checkpoint_path = getattr(self.args, "resume_ckpt", None)
        if checkpoint_path is None:
            return None

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = load_checkpoint_object(checkpoint_path, map_location=self.device)
        load_model_state(self.model, checkpoint)
        print(f"Loaded checkpoint for stage1 training: {checkpoint_path}")
        return checkpoint

    def train(self, setting):
        train_data, _ = self._get_data(flag="TRAIN")
        train_loader = self._build_train_loader(train_data)
        train_eval_loader = self._build_eval_loader(train_data)
        vali_data, _ = self._get_data(flag="VAL")
        vali_loader = self._build_eval_loader(vali_data)
        test_data, _ = self._get_data(flag="TEST")

        print(self._dataset_summary(train_data, "TRAIN"))
        print(self._dataset_summary(vali_data, "VAL"))
        print(self._dataset_summary(test_data, "TEST"))
        self._prepare_metric_logging(setting, train_data, vali_data, test_data)

        checkpoint_dir = self._checkpoint_dir(setting)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        self._select_criterion()
        self._maybe_resume_from_checkpoint()

        best_val_kmeans = -np.inf
        early_stop_counter = 0

        for epoch in range(self.args.train_epochs):
            step_total_losses = []
            step_classification_losses = []
            step_metric_losses = []
            step_mined_triplets = []

            self.model.train()
            epoch_time = time.time()
            for batch_x, label, padding_mask in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                logits, fused_features, embeddings = self._extract_features(batch_x)
                loss, classification_loss, metric_loss, batch_triplets = self._compute_total_loss(
                    logits, embeddings, label
                )

                step_total_losses.append(loss.item())
                step_classification_losses.append(classification_loss)
                step_metric_losses.append(metric_loss)
                step_mined_triplets.append(batch_triplets)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_time_cost}")
            train_step_loss = float(np.average(step_total_losses))
            train_step_cls_loss = float(np.average(step_classification_losses))
            train_step_metric_loss = float(np.average(step_metric_losses))
            train_step_triplets = int(np.sum(step_mined_triplets))

            (
                train_eval_loss,
                train_eval_cls_loss,
                train_eval_metric_loss,
                train_kmeans,
                train_eval_triplets,
            ) = self._evaluate_loader(train_eval_loader)
            (
                val_loss,
                val_cls_loss,
                val_metric_loss,
                val_kmeans,
                val_triplets,
            ) = self._evaluate_loader(vali_loader)
            current_lr = model_optim.param_groups[0]["lr"]

            self._append_metric_row(
                epoch=epoch + 1,
                split="train",
                loss=train_eval_loss,
                classification_loss=train_eval_cls_loss,
                metric_loss=train_eval_metric_loss,
                kmeans_score=train_kmeans,
                mined_triplets=train_eval_triplets,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )
            self._append_metric_row(
                epoch=epoch + 1,
                split="val",
                loss=val_loss,
                classification_loss=val_cls_loss,
                metric_loss=val_metric_loss,
                kmeans_score=val_kmeans,
                mined_triplets=val_triplets,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | "
                f"Train Step Loss: {train_step_loss:.5f}, "
                f"Cls: {train_step_cls_loss:.5f}, Metric: {train_step_metric_loss:.5f}\n"
                f"Train feature results --- Loss: {train_eval_loss:.5f}, Cls: {train_eval_cls_loss:.5f}, "
                f"Metric: {train_eval_metric_loss:.5f}, KMeans: {train_kmeans:.5f}, "
                f"MinedTriplets(step/eval): {train_step_triplets}/{train_eval_triplets}\n"
                f"Validation feature results --- Loss: {val_loss:.5f}, Cls: {val_cls_loss:.5f}, "
                f"Metric: {val_metric_loss:.5f}, KMeans: {val_kmeans:.5f}, "
                f"MinedTriplets: {val_triplets}"
            )

            if val_kmeans > best_val_kmeans + 1e-6:
                best_val_kmeans = val_kmeans
                early_stop_counter = 0
                stage1_model = unwrap_model(self.model)
                checkpoint_payload = build_checkpoint_payload(
                    self.model,
                    optimizer=model_optim,
                    epoch=epoch,
                    best_metric=best_val_kmeans,
                    args_dict=vars(deepcopy(self.args)),
                    extra={
                        "stage": "stage1_feature_training",
                        "backbone_state_dict": stage1_model.backbone.state_dict(),
                    },
                )
                torch.save(checkpoint_payload, checkpoint_dir / "checkpoint.pth")
                print(
                    f"Validation KMeans improved to {best_val_kmeans:.5f}. Saving best feature checkpoint ..."
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
        vali_data, _ = self._get_data(flag="VAL")
        vali_loader = self._build_eval_loader(vali_data)
        test_data, _ = self._get_data(flag="TEST")
        test_loader = self._build_eval_loader(test_data)
        checkpoint_dir = self._checkpoint_dir(setting)

        best_model_path = checkpoint_dir / "checkpoint.pth"
        if best_model_path.exists():
            load_model_state(self.model, best_model_path, map_location=self.device)

        self._select_criterion()
        (
            val_loss,
            val_cls_loss,
            val_metric_loss,
            val_kmeans,
            val_triplets,
        ) = self._evaluate_loader(vali_loader)
        (
            test_loss,
            test_cls_loss,
            test_metric_loss,
            test_kmeans,
            test_triplets,
        ) = self._evaluate_loader(test_loader)

        self._append_metric_row(
            epoch="final",
            split="val_final",
            loss=val_loss,
            classification_loss=val_cls_loss,
            metric_loss=val_metric_loss,
            kmeans_score=val_kmeans,
            mined_triplets=val_triplets,
        )
        self._append_metric_row(
            epoch="final",
            split="test_final",
            loss=test_loss,
            classification_loss=test_cls_loss,
            metric_loss=test_metric_loss,
            kmeans_score=test_kmeans,
            mined_triplets=test_triplets,
        )

        print(
            f"Final validation feature results --- Loss: {val_loss:.5f}, Cls: {val_cls_loss:.5f}, "
            f"Metric: {val_metric_loss:.5f}, KMeans: {val_kmeans:.5f}, MinedTriplets: {val_triplets}\n"
            f"Final test feature results --- Loss: {test_loss:.5f}, Cls: {test_cls_loss:.5f}, "
            f"Metric: {test_metric_loss:.5f}, KMeans: {test_kmeans:.5f}, MinedTriplets: {test_triplets}"
        )

        if not getattr(self.args, "keep_checkpoint", True):
            self.del_weight(str(checkpoint_dir))

        return {"ValKMeans": val_kmeans, "TestKMeans": test_kmeans}

    def del_weight(self, path):
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Model weights deleted....")
