import numpy as np
import pytest
from src.indicators.overlap.overlap_indicators import sma_numba

def test_sma_numba_basic():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    length = 3
    result = sma_numba(arr, length)

    # Expected output: [NaN, NaN, (1+2+3)/3, (2+3+4)/3, (3+4+5)/3]
    #                 [NaN, NaN, 2.0, 3.0, 4.0]

    assert np.isnan(result[0])
    assert np.isnan(result[1])
    np.testing.assert_array_almost_equal(result[2:], [2.0, 3.0, 4.0])

def test_sma_numba_length_greater_than_array():
    arr = np.array([1.0, 2.0])
    length = 5
    result = sma_numba(arr, length)

    # Expected output: [NaN, NaN]
    assert len(result) == 2
    assert np.all(np.isnan(result))

def test_sma_numba_length_one():
    arr = np.array([1.0, 2.0, 3.0])
    length = 1
    result = sma_numba(arr, length)

    # Expected output: identical to input
    np.testing.assert_array_almost_equal(result, arr)

def test_sma_numba_all_same_values():
    arr = np.array([5.0, 5.0, 5.0, 5.0])
    length = 2
    result = sma_numba(arr, length)

    # Expected output: [NaN, 5.0, 5.0, 5.0]
    assert np.isnan(result[0])
    np.testing.assert_array_almost_equal(result[1:], [5.0, 5.0, 5.0])

def test_sma_numba_with_zeros():
    arr = np.array([0.0, 0.0, 0.0])
    length = 2
    result = sma_numba(arr, length)

    # Expected output: [NaN, 0.0, 0.0]
    assert np.isnan(result[0])
    np.testing.assert_array_almost_equal(result[1:], [0.0, 0.0])

def test_sma_numba_with_negative_numbers():
    arr = np.array([-1.0, -2.0, -3.0, -4.0])
    length = 2
    result = sma_numba(arr, length)

    # Expected output: [NaN, -1.5, -2.5, -3.5]
    assert np.isnan(result[0])
    np.testing.assert_array_almost_equal(result[1:], [-1.5, -2.5, -3.5])
