import logging

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate(gt: np.ndarray, pred: np.ndarray, prefix: str = "") -> float:
    """Compute and log Accuracy / Precision / Recall / F-score.

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth labels, shape [N], values in {0, 1}.
    pred : np.ndarray
        Binary predictions, shape [N], values in {0, 1}.
    prefix : str
        Optional label prepended to the log line, e.g. "Edge" or "Hybrid".

    Returns
    -------
    float
        F-score.
    """
    gt   = gt.astype(int)
    pred = pred.astype(int)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, pred, average="binary"
    )
    label = f"[{prefix}] " if prefix else ""
    logging.info(
        "%sAccuracy: %.2f%%  Precision: %.2f%%  Recall: %.2f%%  F-score: %.2f%%",
        label, accuracy * 100, precision * 100, recall * 100, f_score * 100,
    )
    return float(f_score)
