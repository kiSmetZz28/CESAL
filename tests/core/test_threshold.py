import numpy as np
import pytest

from cesal_inference_pipeline.lad_qbat_edge import (
    compute_threshold_from_energy,
    set_thresh_em,
)


def test_set_thresh_em_shape(small_energy):
    labels = set_thresh_em(small_energy.reshape(-1, 1))
    assert labels.shape == (len(small_energy),)


def test_set_thresh_em_n_components(small_energy):
    labels = set_thresh_em(small_energy.reshape(-1, 1), n_components=3)
    unique = np.unique(labels)
    assert len(unique) <= 3


def test_compute_threshold_returns_float(small_energy):
    thresh, normal_ratio, cluster_pcts = compute_threshold_from_energy(small_energy)
    assert isinstance(thresh, float)


def test_normal_ratio_in_range(small_energy):
    _, normal_ratio, _ = compute_threshold_from_energy(small_energy)
    assert 0 < normal_ratio <= 100


def test_cluster_percentages_sum_to_100(small_energy):
    _, _, cluster_pcts = compute_threshold_from_energy(small_energy)
    total = sum(pct for _, pct in cluster_pcts)
    assert abs(total - 100.0) < 1e-4


def test_threshold_within_energy_range(small_energy):
    thresh, _, _ = compute_threshold_from_energy(small_energy)
    assert small_energy.min() <= thresh <= small_energy.max()
