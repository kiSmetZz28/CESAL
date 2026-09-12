import numpy as np
import pytest

from cesal_inference_pipeline.lad_qbat_edge import compute_binary_predictions


def test_all_above_threshold():
    scores = np.array([1.0, 2.0, 3.0])
    preds = compute_binary_predictions(scores, threshold=0.5)
    assert (preds == 1).all()


def test_all_below_threshold():
    scores = np.array([0.1, 0.2, 0.3])
    preds = compute_binary_predictions(scores, threshold=0.5)
    assert (preds == 0).all()


def test_mixed_predictions():
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    preds = compute_binary_predictions(scores, threshold=0.5)
    np.testing.assert_array_equal(preds, [0, 1, 0, 1])


def test_output_shape_is_1d():
    scores = np.random.rand(50)
    preds = compute_binary_predictions(scores, threshold=0.5)
    assert preds.ndim == 1
    assert preds.shape == (50,)


def test_output_dtype_is_int():
    scores = np.array([0.3, 0.7])
    preds = compute_binary_predictions(scores, threshold=0.5)
    assert preds.dtype in (np.int32, np.int64, int)


def test_boundary_equal_to_threshold():
    # strictly greater than, so boundary value should predict 0
    scores = np.array([0.5])
    preds = compute_binary_predictions(scores, threshold=0.5)
    assert preds[0] == 0
