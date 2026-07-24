import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TOPK_VALUES = (1, 3, 5, 10)


def cluster_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_pred.size == y_true.size

    num_labels = max(int(y_pred.max()), int(y_true.max())) + 1
    cost_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
    for idx in range(y_pred.size):
        cost_matrix[y_pred[idx], y_true[idx]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum()) / float(y_pred.size)


def compute_kmeans_score(embeddings, labels, n_clusters, random_state=45):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    pred_labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    ).fit_predict(embeddings)
    return cluster_accuracy(labels, pred_labels)


def ensure_score_matrix(scores):
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return scores


def compute_classification_metrics(scores, labels, topk_values=TOPK_VALUES):
    scores = ensure_score_matrix(scores)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    num_classes = scores.shape[1]

    score_tensor = torch.from_numpy(scores).float()
    label_tensor = torch.from_numpy(labels).long()
    probs = F.softmax(score_tensor, dim=1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()

    max_k = min(max(topk_values), num_classes)
    topk_indices = torch.topk(probs, k=max_k, dim=1).indices
    label_column = label_tensor.view(-1, 1)
    topk_correct = topk_indices.eq(label_column)

    topk_metrics = {}
    for k in topk_values:
        effective_k = min(k, num_classes)
        topk_metrics[f"Top{k}Accuracy"] = float(
            topk_correct[:, :effective_k].any(dim=1).float().mean().item()
        )

    probs_np = probs.cpu().numpy()
    trues_onehot = (
        F.one_hot(label_tensor, num_classes=num_classes).float().cpu().numpy()
    )

    try:
        if num_classes == 2:
            auroc = roc_auc_score(labels, probs_np[:, 1])
        else:
            auroc = roc_auc_score(trues_onehot, probs_np, multi_class="ovr")
    except ValueError:
        auroc = np.nan

    try:
        if num_classes == 2:
            auprc = average_precision_score(labels, probs_np[:, 1])
        else:
            auprc = average_precision_score(trues_onehot, probs_np, average="macro")
    except ValueError:
        auprc = np.nan

    return {
        "Accuracy": accuracy_score(labels, predictions),
        **topk_metrics,
        "Precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "F1": f1_score(labels, predictions, average="macro", zero_division=0),
        "AUROC": auroc,
        "AUPRC": auprc,
    }


def format_topk_summary(metrics_dict, topk_values=TOPK_VALUES):
    return ", ".join(
        f"Top{k}: {metrics_dict[f'Top{k}Accuracy']:.5f}" for k in topk_values
    )
