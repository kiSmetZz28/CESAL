import numpy as np
import pytest

from cesal_core.utils.voting import ensemble_method


def test_majority_all_agree():
    preds = np.ones((10, 3), dtype=int)
    result = ensemble_method('majority', preds)
    assert (result == 1).all()


def test_majority_none_agree():
    preds = np.zeros((10, 3), dtype=int)
    result = ensemble_method('majority', preds)
    assert (result == 0).all()


def test_majority_split():
    # 2 out of 3 models predict 1 -> majority vote = 1
    preds = np.array([[1, 1, 0]] * 5)
    result = ensemble_method('majority', preds)
    assert (result == 1).all()


def test_at_least_one():
    preds = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    result = ensemble_method('at least one', preds)
    np.testing.assert_array_equal(result, [1, 0, 1])


def test_consensus():
    preds = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 0]])
    result = ensemble_method('consensus', preds)
    np.testing.assert_array_equal(result, [1, 0, 0])


def test_unknown_method_raises():
    preds = np.ones((5, 2), dtype=int)
    with pytest.raises(ValueError, match="unknown"):
        ensemble_method('unknown', preds)


def test_shape_preserved(binary_preds):
    result = ensemble_method('majority', binary_preds)
    assert result.shape == (binary_preds.shape[0],)
