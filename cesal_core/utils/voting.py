import numpy as np


def ensemble_method(method: str, ensemble_data: np.ndarray) -> np.ndarray:
    """Combine per-model 0/1 predictions [n_samples, n_models] into [n_samples] by
    'majority', 'at least one' or 'consensus' voting (ValueError otherwise)."""

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
