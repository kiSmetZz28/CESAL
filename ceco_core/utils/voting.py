import numpy as np


def ensemble_method(method: str, ensemble_data: np.ndarray) -> np.ndarray:
    """Apply ensemble voting to a matrix of per-model binary predictions.

    Parameters
    ----------
    method : {'majority', 'at least one', 'consensus'}
        Voting strategy.
    ensemble_data : np.ndarray
        Shape [n_samples, n_models], dtype int (0/1).

    Returns
    -------
    np.ndarray of shape [n_samples] with final 0/1 predictions.

    Raises
    ------
    ValueError
        If method is not one of the recognised strategies.
    """

    if method == 'majority':
        return (ensemble_data.sum(axis=1) >= (ensemble_data.shape[1] // 2 + 1)).astype(int)
    if method == 'at least one':
        return np.any(ensemble_data == 1, axis=1).astype(int)
    if method == 'consensus':
        return np.all(ensemble_data == 1, axis=1).astype(int)
    raise ValueError(
        f"Unknown voting method: {method!r}. "
        "Expected one of: 'majority', 'at least one', 'consensus'."
    )
