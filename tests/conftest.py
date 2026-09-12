"""Shared pytest fixtures for cloud and edge test suites."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_energy(rng):
    """Small 1-D energy array for threshold tests."""
    return rng.normal(loc=0.5, scale=0.2, size=200).astype(np.float32)


@pytest.fixture
def binary_preds(rng):
    """[n_samples, n_models] binary prediction matrix."""
    return rng.integers(0, 2, size=(100, 3)).astype(int)
