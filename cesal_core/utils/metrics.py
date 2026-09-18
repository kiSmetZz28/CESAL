import logging
from typing import NamedTuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from cesal_core.utils import steps


class Scores(NamedTuple):
    """Detection scores as percentages (0-100)."""
    accuracy: float
    precision: float
    recall: float
    f_score: float


def evaluate(gt: np.ndarray, pred: np.ndarray, prefix: str = "",
             register: bool = True, level: int = logging.INFO) -> Scores:
    """Compute, log and register Accuracy / Precision / Recall / F-score.

    When a run is active (see :mod:`cesal_core.utils.steps`) the scores are also
    registered with it, so the end-of-run summary can show every scored stage
    side by side without the caller passing them along.

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth labels, shape [N], values in {0, 1}.
    pred : np.ndarray
        Binary predictions, shape [N], values in {0, 1}.
    prefix : str
        Optional label for the result, e.g. "Edge" or "Hybrid".
    register : bool
        Add the scores to the run summary. Set False for intermediate results —
        the ensemble sweep scores hundreds of partial ensembles, and every one
        of those in the summary would bury the handful that matter.
    level : int
        Level to log the score line at; DEBUG keeps an intermediate result in
        the log file without printing it.

    Returns
    -------
    Scores
        Accuracy / precision / recall / F-score, all as percentages.
    """
    gt   = gt.astype(int)
    pred = pred.astype(int)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, pred, average="binary"
    )
    scores = Scores(accuracy * 100, precision * 100, recall * 100, f_score * 100)

    run = steps.current_run()
    # Indent inside a run so the line sits within its step block, one level
    # deeper while a phase is open. The wording after the indent is deliberately
    # unchanged: dashboard/app.py's _parse_log_metrics and the front-end both
    # regex-match it, when replaying a saved log file and when following a live
    # run respectively.
    indent = " " * steps.body_indent()
    label = f"[{prefix}] " if prefix else ""
    logging.log(
        level,
        "%s%sAccuracy: %.2f%%  Precision: %.2f%%  Recall: %.2f%%  F-score: %.2f%%",
        indent, label, scores.accuracy, scores.precision, scores.recall, scores.f_score,
    )
    if run is not None and register:
        run.metric(prefix or "Result", *scores)

    return scores
