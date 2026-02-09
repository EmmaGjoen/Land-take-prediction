"""Utility metrics for binary segmentation tasks.

Provides confusion matrix computation and common derived metrics.
"""
from typing import Tuple, Dict

import torch


def compute_confusion_binary(y_pred: torch.Tensor, y_true: torch.Tensor, positive_class: int = 1) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix for binary classification.
    y_pred, y_true: (B, H, W) with 0/1 labels (torch tensors)

    Returns TP, FP, TN, FN as Python integers.
    """
    y_pred = (y_pred == positive_class)
    y_true = (y_true == positive_class)

    tp = (y_pred & y_true).sum().item()
    fp = (y_pred & ~y_true).sum().item()
    tn = (~y_pred & ~y_true).sum().item()
    fn = (~y_pred & y_true).sum().item()
    return tp, fp, tn, fn


def compute_metrics_from_confusion(tp: int, fp: int, tn: int, fn: int, eps: float = 1e-8) -> Dict[str, float]:
    """
    Compute metrics from confusion matrix values.
    Returns: dict with accuracy, precision, recall, f1, iou
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }
