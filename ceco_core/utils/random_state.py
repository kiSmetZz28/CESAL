import logging
import random
from itertools import product
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

# Resolve relative config paths from the project root (two levels above this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_random_state(
    config_path: str,
    num_epochs: int,
    k: int,
    e_layer_num: int,
    batch_size: int,
) -> int:
    """Return a unique, reproducible random seed for the given hyperparameter combination.

    Seeds are drawn once (random.seed(42)) from the full Cartesian product defined
    in the YAML config, so each combination always maps to the same seed regardless
    of the order models are trained.

    Parameters
    ----------
    config_path : str
        Path to the BAT training YAML config (relative to project root or absolute).
    num_epochs, k, e_layer_num, batch_size : int
        The specific hyperparameter values for this model.

    Returns
    -------
    int
        A unique seed in [10, 301).
    """
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / config_path

    with open(resolved, 'r') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    try:
        num_epochs_list = cfg['num_epochs']
        k_list = cfg['k']
        e_layer_num_list = cfg['e_layer_num']
        batch_size_list = cfg['batch_size']
    except KeyError as exc:
        raise KeyError(f"Missing required key in config: {exc}") from exc

    all_combinations = list(product(num_epochs_list, k_list, e_layer_num_list, batch_size_list))

    random.seed(42)
    unique_random_states = random.sample(range(10, 301), len(all_combinations))

    combo_to_seed: Dict[Tuple[int, int, int, int], int] = {
        combo: seed for combo, seed in zip(all_combinations, unique_random_states)
    }

    key = (num_epochs, k, e_layer_num, batch_size)
    if key not in combo_to_seed:
        raise ValueError(f"Invalid hyperparameter combination: {key}")

    logging.debug("Combination %s -> random seed %d", key, combo_to_seed[key])
    return combo_to_seed[key]
