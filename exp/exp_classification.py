from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
import csv
import json
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

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
        losses = F.relu(
            hardest_positive[valid] - hardest_negative[valid] + self.margin
        )
        return losses.mean()


class Exp_Classification(Exp_Basic):
    METRIC_FIELDNAMES = [
        "epoch",
        "split",
        "loss",
        "opt_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "learning_rate",
        "epoch_time_sec",
        "steps",
    ]

    def __init__(self, args):
        super().__init__(args)
        self.metrics_log_path = None
        self.split_summary_path = None
        self.run_params_path = None
        self.triplet_criterion = None

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
        payload = {
            "split": split_name,
            "samples": len(dataset),
            "seq_len": self._dataset_seq_len(dataset),
            "feat_dim": self._dataset_feature_dim(dataset),
            "num_class": self._dataset_num_class(dataset),
        }

        ids_attr = {
            "TRAIN": "train_ids",
            "VAL": "val_ids",
            "TEST": "test_ids",
        }.get(split_name)
        if ids_attr and hasattr(dataset, ids_attr):
            payload["ids"] = list(getattr(dataset, ids_attr))
        return payload

    def _checkpoint_dir(self, setting):
        return Path("./checkpoints") / self.args.model / setting

    def _log_dir(self, setting):
        log_root = getattr(self.args, "log_dir", None)
        if log_root is None:
            return self._checkpoint_dir(setting)
        return Path(log_root) / self.args.data / setting

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
        metrics_dict,
        learning_rate=None,
        epoch_time=None,
        steps=None,
        opt_loss=None,
    ):
        if self.metrics_log_path is None:
            return

        row = {
            "epoch": epoch,
            "split": split,
            "loss": float(loss),
            "opt_loss": "" if opt_loss is None else float(opt_loss),
            "accuracy": float(metrics_dict["Accuracy"]),
            "precision": float(metrics_dict["Precision"]),
            "recall": float(metrics_dict["Recall"]),
            "f1": float(metrics_dict["F1"]),
            "auroc": float(metrics_dict["AUROC"]),
            "auprc": float(metrics_dict["AUPRC"]),
            "learning_rate": "" if learning_rate is None else float(learning_rate),
            "epoch_time_sec": "" if epoch_time is None else float(epoch_time),
            "steps": "" if steps is None else int(steps),
        }

        with self.metrics_log_path.open("a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.METRIC_FIELDNAMES)
            writer.writerow(row)

    def _compute_metrics(self, preds, trues):
        probs = F.softmax(
            preds, dim=1
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            F.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        try:
            auroc = roc_auc_score(trues_onehot, probs, multi_class="ovr")
        except ValueError:
            auroc = np.nan

        try:
            auprc = average_precision_score(trues_onehot, probs, average="macro")
        except ValueError:
            auprc = np.nan

        return {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(
                trues, predictions, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                trues, predictions, average="macro", zero_division=0
            ),
            "F1": f1_score(trues, predictions, average="macro", zero_division=0),
            "AUROC": auroc,
            "AUPRC": auprc,
        }

    def _use_triplet_loss(self):
        return self.args.loss == "ce_triplet"

    def _forward_model(self, batch_x):
        if self._use_triplet_loss():
            return self.model(batch_x, return_features=True)
        return self.model(batch_x)

    def _compute_objective(self, model_outputs, labels, ce_criterion):
        if self._use_triplet_loss():
            logits, features = model_outputs
            ce_loss = ce_criterion(logits, labels.long())
            normalized_features = F.normalize(features, p=2, dim=1)
            triplet_loss = self.triplet_criterion(normalized_features, labels.long())
            total_loss = ce_loss + self.args.triplet_weight * triplet_loss
            return total_loss, logits

        logits = model_outputs
        total_loss = ce_criterion(logits, labels.long())
        return total_loss, logits

    def _evaluate_loader(self, data_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                model_outputs = self._forward_model(batch_x)
                loss, logits = self._compute_objective(
                    model_outputs, label, criterion
                )
                pred = logits.detach().cpu()
                total_loss.append(loss.item())

                preds.append(pred)
                trues.append(label.detach().cpu())

        total_loss = float(np.average(total_loss))
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        metrics_dict = self._compute_metrics(preds, trues)
        self.model.train()
        return total_loss, metrics_dict

    def _build_model(self):
        # model input depends on data
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = self._dataset_seq_len(test_data)  # redefine seq_len
        self.args.pred_len = 0
        self.args.enc_in = self._dataset_feature_dim(test_data)  # redefine enc_in
        self.args.num_class = self._dataset_num_class(test_data)
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        self.triplet_criterion = None
        if self.args.loss == "ce":
            return nn.CrossEntropyLoss()
        if self.args.loss == "ce_triplet":
            self.triplet_criterion = BatchHardTripletLoss(
                margin=self.args.triplet_margin
            )
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unsupported loss type: {self.args.loss}")

    def vali(self, vali_data, vali_loader, criterion):
        return self._evaluate_loader(vali_loader, criterion)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(self._dataset_summary(train_data, "TRAIN"))
        print(self._dataset_summary(vali_data, "VAL"))
        print(self._dataset_summary(test_data, "TEST"))
        self._prepare_metric_logging(setting, train_data, vali_data, test_data)

        checkpoint_dir = self._checkpoint_dir(setting)
        path = str(checkpoint_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                model_outputs = self._forward_model(batch_x)
                loss, logits = self._compute_objective(
                    model_outputs, label, criterion
                )
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time_cost))
            train_loss = float(np.average(train_loss))
            train_eval_loss, train_metrics_dict = self.vali(
                train_data, train_loader, criterion
            )
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            current_lr = model_optim.param_groups[0]["lr"]

            self._append_metric_row(
                epoch=epoch + 1,
                split="train",
                loss=train_eval_loss,
                metrics_dict=train_metrics_dict,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
                opt_loss=train_loss,
            )
            self._append_metric_row(
                epoch=epoch + 1,
                split="val",
                loss=vali_loss,
                metrics_dict=val_metrics_dict,
                learning_rate=current_lr,
                epoch_time=epoch_time_cost,
                steps=train_steps,
            )

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Step Loss: {train_loss:.5f}\n"
                f"Train results --- Loss: {train_eval_loss:.5f}, "
                f"Accuracy: {train_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {train_metrics_dict['Precision']:.5f}, "
                f"Recall: {train_metrics_dict['Recall']:.5f}, "
                f"F1: {train_metrics_dict['F1']:.5f}, "
                f"AUROC: {train_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {train_metrics_dict['AUPRC']:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}"
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = checkpoint_dir / "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        
        checkpoint_dir = self._checkpoint_dir(setting)
        path = str(checkpoint_dir)
        if test:
            print("loading model")
            model_path = checkpoint_dir / "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            self.model.load_state_dict(torch.load(model_path))
            
        # # Uncomment below code for save space on device
        self.del_weight(path)

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)
        self._append_metric_row(
            epoch="final",
            split="val_final",
            loss=vali_loss,
            metrics_dict=val_metrics_dict,
        )
        self._append_metric_row(
            epoch="final",
            split="test_final",
            loss=test_loss,
            metrics_dict=test_metrics_dict,
        )

        print(
            f"Final validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Final test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        return test_metrics_dict
    
    def del_weight(self, path):
        if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
            os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
            print('Model weights deleted....')
