import numpy as np
import pytest

from cesal_core.models.EMAT import EMAT, fit_gmm


def test_fit_gmm_shape(small_energy):
    labels = fit_gmm(small_energy.reshape(-1, 1))
    assert labels.shape == (len(small_energy),)


def test_fit_gmm_n_components(small_energy):
    labels = fit_gmm(small_energy.reshape(-1, 1), n_components=3)
    unique = np.unique(labels)
    assert len(unique) <= 3


def test_fit_gmm_runs_on_one_thread(small_energy, monkeypatch):
    """The fit depends on the BLAS/OpenMP thread count, so it must be pinned to one
    thread: that is how the released thresholds were fitted."""
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_info

    seen = []
    original = GaussianMixture.fit

    def recording_fit(self, X, y=None):
        seen.extend(pool['num_threads'] for pool in threadpool_info())
        return original(self, X, y)

    monkeypatch.setattr(GaussianMixture, 'fit', recording_fit)
    fit_gmm(small_energy.reshape(-1, 1))
    assert seen and set(seen) == {1}


def test_fit_threshold_returns_float(small_energy):
    thresh, normal_ratio, cluster_pcts = EMAT.fit_threshold(small_energy)
    assert isinstance(thresh, float)


def test_normal_ratio_in_range(small_energy):
    _, normal_ratio, _ = EMAT.fit_threshold(small_energy)
    assert 0 < normal_ratio <= 100


def test_cluster_percentages_sum_to_100(small_energy):
    _, _, cluster_pcts = EMAT.fit_threshold(small_energy)
    total = sum(pct for _, pct in cluster_pcts)
    assert abs(total - 100.0) < 1e-4


def test_threshold_within_energy_range(small_energy):
    thresh, _, _ = EMAT.fit_threshold(small_energy)
    assert small_energy.min() <= thresh <= small_energy.max()
