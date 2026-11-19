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
    """Compute and log accuracy / precision / recall / F-score (percentages).

    Inside a run the scores are also registered for the end-of-run summary, unless
    ``register`` is False (intermediate results such as partial ensembles). ``level``
    is the log level of the score line; DEBUG keeps it out of the terminal.
    """
    gt   = gt.astype(int)
    pred = pred.astype(int)

    accuracy = accuracy_score(gt, pred)
    # zero_division=0 is already sklearn's result; stating it silences a warning that
    # otherwise fires once per learner that flags nothing.
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0
    )
    scores = Scores(accuracy * 100, precision * 100, recall * 100, f_score * 100)

    run = steps.current_run()
    # Indented to sit inside the step block. Keep the wording: dashboard/app.py's
    # _parse_log_metrics and the front end regex-match this line.
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
