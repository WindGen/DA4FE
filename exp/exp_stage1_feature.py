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
from torch import optim
from torch.utils.data import BatchSampler, DataLoader

from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn
from exp.exp_basic import Exp_Basic
from utils.checkpointing import build_checkpoint_payload, load_checkpoint_object, load_model_state
from utils.stage_metrics import compute_kmeans_score
from utils.tools import adjust_learning_rate

warnings.filterwarnings("ignore")


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        if features.ndim != 2:
            raise ValueError("Triplet features must be a 2D tensor")

        labels = labels.reshape(-1)
        batch_size = labels.size(0)
        if batch_size < 2:
            return features.new_zeros(())

        distances = torch.cdist(features, features, p=2)
        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        diag_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
        positive_mask = same_label & ~diag_mask
        negative_mask = ~same_label

        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return features.new_zeros(())

        hardest_positive = distances.masked_fill(
            ~positive_mask, float("-inf")
        ).max(dim=1).values
        hardest_negative = distances.masked_fill(
            ~negative_mask, float("inf")
        ).min(dim=1).values
        losses = torch.relu(
            hardest_positive[valid] - hardest_negative[valid] + self.margin
        )
        return losses.mean()


class TripletSemiHardLoss(nn.Module):
    def masked_maximum(self, data, mask, dim=1):
        axis_minimums = torch.min(data, dim, keepdim=True).values
        return (
            torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values
            + axis_minimums
        )

    def masked_minimum(self, data, mask, dim=1):
        axis_maximums = torch.max(data, dim, keepdim=True).values
        return (
            torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values
            + axis_maximums
        )

    def pairwise_distance(self, embeddings, squared=True):
        pairwise_distances_squared = (
            torch.sum(embeddings**2, dim=1, keepdim=True)
            + torch.sum(embeddings.t() ** 2, dim=0, keepdim=True)
            - 2.0 * torch.matmul(embeddings, embeddings.t())
        )

        error_mask = pairwise_distances_squared <= 0.0
        if squared:
            pairwise_distances = pairwise_distances_squared.clamp(min=0)
        else:
            pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

        pairwise_distances = torch.mul(pairwise_distances, ~error_mask)
        num_data = embeddings.shape[0]
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(
            torch.ones([num_data], device=embeddings.device)
        )
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
        return pairwise_distances

    def forward(self, embeddings, target, margin=1.0, squared=True):
        labels = target.int().unsqueeze(-1)
        pdist_matrix = self.pairwise_distance(embeddings, squared=squared)
        adjacency = labels == torch.transpose(labels, 0, 1)
        adjacency_not = ~adjacency
        batch_size = labels.shape[0]

        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])
        mask = adjacency_not.repeat([batch_size, 1]) & (
            pdist_matrix_tile
            > torch.reshape(torch.transpose(pdist_matrix, 0, 1), [-1, 1])
        )
        mask_final = torch.reshape(
            torch.sum(mask.float(), 1, keepdim=True) > 0.0, [batch_size, batch_size]
        )
        mask_final = torch.transpose(mask_final, 0, 1)

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
        )
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        negatives_inside = self.masked_maximum(
            pdist_matrix, adjacency_not
        ).repeat([1, batch_size])
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)
        mask_positives = adjacency.float() - torch.diag(
            torch.ones([batch_size], device=embeddings.device)
        )
        num_positives = torch.sum(mask_positives)
        if num_positives.item() == 0:
            return embeddings.new_zeros(())
        return torch.div(
            torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)),
            num_positives,
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
        "kmeans_score",
        "learning_rate",
        "epoch_time_sec",
        "steps",
    ]

    def __init__(self, args):
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
        kmeans_score,
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
            "kmeans_score": float(kmeans_score),
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

    def _build_model(self):
        test_data, _ = self._get_data(flag="TEST")
        self.args.seq_len = self._dataset_seq_len(test_data)
        self.args.pred_len = 0
        self.args.enc_in = self._dataset_feature_dim(test_data)
        self.args.num_class = self._dataset_num_class(test_data)
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        return data_provider(self.args, flag)

    def _build_train_loader(self, train_data):
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
        if self.args.stage1_triplet_type == "semihard":
            return TripletSemiHardLoss()
        if self.args.stage1_triplet_type == "batch_hard":
            return BatchHardTripletLoss(margin=self.args.stage1_triplet_margin)
        raise ValueError(
            f"Unsupported stage1_triplet_type: {self.args.stage1_triplet_type}"
        )

    def _extract_features(self, batch_x):
        model_outputs = self.model(batch_x, return_features=True)
        _, features = model_outputs
        return nn.functional.normalize(features, p=2, dim=1)

    def _compute_triplet_loss(self, criterion, features, labels):
        if isinstance(criterion, TripletSemiHardLoss):
            return criterion(features, labels.long(), margin=self.args.stage1_triplet_margin)
        return criterion(features, labels.long())

    def _evaluate_loader(self, data_loader, criterion):
        total_loss = []
        feature_rows = []
        label_rows = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                features = self._extract_features(batch_x)
                loss = self._compute_triplet_loss(criterion, features, label)
                total_loss.append(loss.item())
                feature_rows.append(features.cpu().numpy())
                label_rows.append(label.detach().cpu().numpy())

        total_loss = float(np.average(total_loss))
        embeddings = np.concatenate(feature_rows, axis=0)
        labels = np.concatenate(label_rows, axis=0)
        kmeans_score = compute_kmeans_score(
            embeddings,
            labels,
            n_clusters=self._feature_kmeans_clusters(),
            random_state=self.args.seed,
        )
        self.model.train()
        return total_loss, kmeans_score

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
        criterion = self._select_criterion()
        self._maybe_resume_from_checkpoint()

        best_val_kmeans = -np.inf
        early_stop_counter = 0

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for batch_x, label, padding_mask in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                features = self._extract_features(batch_x)
                loss = self._compute_triplet_loss(criterion, features, label)
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_time_cost}")
            train_step_loss = float(np.average(train_loss))
            train_eval_loss, train_kmeans = self._evaluate_loader(train_eval_loader, criterion)
            val_loss, val_kmeans = self._evaluate_loader(vali_loader, criterion)
            current_lr = model_optim.param_groups[0]["lr"]

            self._append_metric_row(
                epoch=epoch + 1,
                split="train",
                loss=train_eval_loss,
                kmeans_score=train_kmeans,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )
            self._append_metric_row(
                epoch=epoch + 1,
                split="val",
                loss=val_loss,
                kmeans_score=val_kmeans,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Step Loss: {train_step_loss:.5f}\n"
                f"Train feature results --- Loss: {train_eval_loss:.5f}, KMeans: {train_kmeans:.5f}\n"
                f"Validation feature results --- Loss: {val_loss:.5f}, KMeans: {val_kmeans:.5f}"
            )

            if val_kmeans > best_val_kmeans + 1e-6:
                best_val_kmeans = val_kmeans
                early_stop_counter = 0
                checkpoint_payload = build_checkpoint_payload(
                    self.model,
                    optimizer=model_optim,
                    epoch=epoch,
                    best_metric=best_val_kmeans,
                    args_dict=vars(deepcopy(self.args)),
                    extra={"stage": "stage1_feature_training"},
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

        criterion = self._select_criterion()
        val_loss, val_kmeans = self._evaluate_loader(vali_loader, criterion)
        test_loss, test_kmeans = self._evaluate_loader(test_loader, criterion)

        self._append_metric_row(
            epoch="final",
            split="val_final",
            loss=val_loss,
            kmeans_score=val_kmeans,
        )
        self._append_metric_row(
            epoch="final",
            split="test_final",
            loss=test_loss,
            kmeans_score=test_kmeans,
        )

        print(
            f"Final validation feature results --- Loss: {val_loss:.5f}, KMeans: {val_kmeans:.5f}\n"
            f"Final test feature results --- Loss: {test_loss:.5f}, KMeans: {test_kmeans:.5f}"
        )

        if not getattr(self.args, "keep_checkpoint", True):
            self.del_weight(str(checkpoint_dir))

        return {"ValKMeans": val_kmeans, "TestKMeans": test_kmeans}

    def del_weight(self, path):
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Model weights deleted....")
