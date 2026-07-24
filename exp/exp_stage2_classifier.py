from copy import deepcopy
import csv
import json
from pathlib import Path
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.checkpointing import load_model_state
from utils.stage_metrics import compute_classification_metrics, format_topk_summary

warnings.filterwarnings("ignore")


class Exp_Stage2_Classifier(Exp_Basic):
    METRIC_FIELDNAMES = [
        "head",
        "split",
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
    ]
    REFERENCE_HEADS = ("linear_svm", "rbf_svm", "poly_svm")
    AVAILABLE_HEADS = ("linear_svm", "rbf_svm", "poly_svm", "logreg", "mlp")

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

    def _dataset_summary_payload(self, dataset, split_name):
        return {
            "split": split_name,
            "samples": len(dataset),
            "seq_len": self._dataset_seq_len(dataset),
            "feat_dim": self._dataset_feature_dim(dataset),
            "num_class": self._dataset_num_class(dataset),
        }

    def _dataset_summary(self, dataset, split_name):
        payload = self._dataset_summary_payload(dataset, split_name)
        return (
            f"{split_name}: samples={payload['samples']}, seq_len={payload['seq_len']}, "
            f"feat_dim={payload['feat_dim']}, num_class={payload['num_class']}"
        )

    def _checkpoint_dir(self, setting):
        return Path("./checkpoints") / self.args.model / "stage2" / setting

    def _log_dir(self, setting):
        log_root = getattr(self.args, "log_dir", None)
        if log_root is None:
            return self._checkpoint_dir(setting)
        return Path(log_root) / self.args.data / "stage2" / setting

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
            "stage": "stage2_classifier_training",
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

    def _append_metric_row(self, head_name, split, metrics_dict):
        if self.metrics_log_path is None:
            return

        row = {
            "head": head_name,
            "split": split,
            "accuracy": float(metrics_dict["Accuracy"]),
            "top1_accuracy": float(metrics_dict["Top1Accuracy"]),
            "top3_accuracy": float(metrics_dict["Top3Accuracy"]),
            "top5_accuracy": float(metrics_dict["Top5Accuracy"]),
            "top10_accuracy": float(metrics_dict["Top10Accuracy"]),
            "precision": float(metrics_dict["Precision"]),
            "recall": float(metrics_dict["Recall"]),
            "f1": float(metrics_dict["F1"]),
            "auroc": float(metrics_dict["AUROC"]),
            "auprc": float(metrics_dict["AUPRC"]),
        }
        with self.metrics_log_path.open("a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.METRIC_FIELDNAMES)
            writer.writerow(row)

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

    def _extract_split_features(self, data_loader):
        features = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                model_outputs = self.model(batch_x, return_features=True)
                _, fused = model_outputs
                fused = nn.functional.normalize(fused, p=2, dim=1)
                features.append(fused.cpu().numpy())
                labels.append(label.cpu().numpy())
        self.model.train()
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def _resolve_heads(self):
        heads_arg = getattr(self.args, "stage2_classifier_heads", "linear_svm").strip()
        if heads_arg.lower() == "reference":
            return list(self.REFERENCE_HEADS)
        if heads_arg.lower() == "all":
            return list(self.AVAILABLE_HEADS)

        heads = [item.strip() for item in heads_arg.split(",") if item.strip()]
        invalid = [head for head in heads if head not in self.AVAILABLE_HEADS]
        if invalid:
            raise ValueError(f"Unsupported stage2 classifier heads: {invalid}")
        return heads

    def _build_head(self, head_name):
        if head_name == "linear_svm":
            return LinearSVC(
                C=self.args.stage2_svm_c,
                multi_class="ovr",
                random_state=self.args.seed,
                max_iter=self.args.stage2_max_iter,
            )
        if head_name == "rbf_svm":
            return SVC(
                kernel="rbf",
                C=self.args.stage2_svm_c,
                gamma=self.args.stage2_svm_gamma,
                decision_function_shape="ovr",
                random_state=self.args.seed,
                max_iter=self.args.stage2_max_iter,
            )
        if head_name == "poly_svm":
            return SVC(
                kernel="poly",
                degree=self.args.stage2_svm_degree,
                C=self.args.stage2_svm_c,
                gamma=self.args.stage2_svm_gamma,
                decision_function_shape="ovr",
                random_state=self.args.seed,
                max_iter=self.args.stage2_max_iter,
            )
        if head_name == "logreg":
            return LogisticRegression(
                C=self.args.stage2_logreg_c,
                max_iter=self.args.stage2_max_iter,
                multi_class="ovr",
                random_state=self.args.seed,
            )
        if head_name == "mlp":
            hidden_dims = tuple(
                int(item)
                for item in str(self.args.stage2_mlp_hidden_dims).split(",")
                if str(item).strip()
            )
            return MLPClassifier(
                hidden_layer_sizes=hidden_dims,
                alpha=self.args.stage2_mlp_alpha,
                learning_rate_init=self.args.stage2_mlp_lr,
                max_iter=self.args.stage2_max_iter,
                random_state=self.args.seed,
            )
        raise ValueError(f"Unknown stage2 head: {head_name}")

    def _score_head(self, head, features):
        if hasattr(head, "predict_proba"):
            try:
                return head.predict_proba(features)
            except Exception:
                pass
        if hasattr(head, "decision_function"):
            return head.decision_function(features)

        predictions = head.predict(features)
        scores = np.zeros((predictions.shape[0], self.args.num_class), dtype=np.float32)
        scores[np.arange(predictions.shape[0]), predictions.astype(int)] = 1.0
        return scores

    def _maybe_standardize(self, train_x, val_x, test_x):
        if not getattr(self.args, "stage2_standardize_features", True):
            return train_x, val_x, test_x, None

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        val_x_scaled = scaler.transform(val_x)
        test_x_scaled = scaler.transform(test_x)
        return train_x_scaled, val_x_scaled, test_x_scaled, scaler

    def train(self, setting):
        if not getattr(self.args, "resume_ckpt", None):
            raise ValueError("stage2 requires --resume_ckpt pointing to a trained stage1 checkpoint")

        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        print(self._dataset_summary(train_data, "TRAIN"))
        print(self._dataset_summary(vali_data, "VAL"))
        print(self._dataset_summary(test_data, "TEST"))
        self._prepare_metric_logging(setting, train_data, vali_data, test_data)

        load_model_state(self.model, self.args.resume_ckpt, map_location=self.device)
        print(f"Loaded stage1 feature checkpoint: {self.args.resume_ckpt}")

        train_x, train_y = self._extract_split_features(train_loader)
        val_x, val_y = self._extract_split_features(vali_loader)
        test_x, test_y = self._extract_split_features(test_loader)

        train_x, val_x, test_x, scaler = self._maybe_standardize(train_x, val_x, test_x)

        checkpoint_dir = self._checkpoint_dir(setting)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        selected_heads = self._resolve_heads()
        all_results = {}

        for head_name in selected_heads:
            head = self._build_head(head_name)
            head.fit(train_x, train_y)

            artifact = {"head": head, "scaler": scaler}
            with (checkpoint_dir / f"{head_name}.pkl").open("wb") as file_obj:
                pickle.dump(artifact, file_obj)

            split_payloads = {
                "train": (train_x, train_y),
                "val": (val_x, val_y),
                "test": (test_x, test_y),
            }
            head_results = {}
            for split_name, (split_x, split_y) in split_payloads.items():
                scores = self._score_head(head, split_x)
                metrics_dict = compute_classification_metrics(scores, split_y)
                self._append_metric_row(head_name, split_name, metrics_dict)
                head_results[split_name] = metrics_dict

                print(
                    f"[{head_name}][{split_name}] "
                    f"Accuracy: {metrics_dict['Accuracy']:.5f}, "
                    f"{format_topk_summary(metrics_dict)}, "
                    f"Precision: {metrics_dict['Precision']:.5f}, "
                    f"Recall: {metrics_dict['Recall']:.5f}, "
                    f"F1: {metrics_dict['F1']:.5f}, "
                    f"AUROC: {metrics_dict['AUROC']:.5f}, "
                    f"AUPRC: {metrics_dict['AUPRC']:.5f}"
                )

            all_results[head_name] = head_results

        return all_results
