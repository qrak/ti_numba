import numpy as np
import pandas as pd
import pytest

from src.indicators.overlap.overlap_indicators import sma_numba, ema_numba, ewma_numba

def test_sma_numba_basic():
    data = np.arange(1, 11, dtype=np.float64)  # 1 to 10
    length = 3
    result = sma_numba(data, length)

    # Expected results using pandas
    expected = pd.Series(data).rolling(window=length).mean().values

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8, equal_nan=True)

def test_sma_numba_length_greater_than_data():
    data = np.arange(1, 6, dtype=np.float64)
    length = 10
    result = sma_numba(data, length)

    expected = np.full(len(data), np.nan)
    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_sma_numba_with_nan():
    # The current implementation of sma_numba does not handle NaNs explicitly.
    # If there are NaNs inside the window, window_sum becomes NaN and all subsequent values become NaN.
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    length = 2
    result = sma_numba(data, length)

    expected = np.array([np.nan, 1.5, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_sma_numba_empty():
    data = np.array([], dtype=np.float64)
    length = 3
    result = sma_numba(data, length)
    expected = np.array([])
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_ema_numba_basic():
    data = np.arange(1, 11, dtype=np.float64)
    length = 3
    result = ema_numba(data, length)

    # Pandas EWM with span=length, adjust=False usually matches traditional EMA
    expected = pd.Series(data).ewm(span=length, adjust=False).mean().values

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

def test_ema_numba_with_leading_nans():
    data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    length = 3
    result = ema_numba(data, length)

    # first_valid is 2 (val 1.0)
    # the rest is EMA from there
    expected_valid = pd.Series([1.0, 2.0, 3.0, 4.0]).ewm(span=length, adjust=False).mean().values
    expected = np.concatenate(([np.nan, np.nan], expected_valid))

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8, equal_nan=True)

def test_ema_numba_all_nans():
    data = np.array([np.nan, np.nan, np.nan])
    length = 3
    result = ema_numba(data, length)
    expected = np.array([np.nan, np.nan, np.nan])
    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_ema_numba_with_nans_in_middle():
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    length = 3
    result = ema_numba(data, length)

    # the function says:
    # if np.isnan(arr[i]): ema_arr[i] = ema_arr[i-1]
    # else: ema_arr[i] = ((arr[i] - ema_arr[i - 1]) * multiplier) + ema_arr[i - 1]
    expected = np.zeros(5)
    expected[0] = 1.0
    expected[1] = (2.0 - 1.0) * (2/4) + 1.0 # 1.5
    expected[2] = expected[1] # 1.5
    expected[3] = (4.0 - 1.5) * (2/4) + 1.5 # 2.75
    expected[4] = (5.0 - 2.75) * (2/4) + 2.75 # 3.875

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

def test_ema_numba_empty():
    data = np.array([], dtype=np.float64)
    length = 3
    result = ema_numba(data, length)
    expected = np.array([])
    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_ewma_numba_basic():
    data = np.arange(1, 11, dtype=np.float64)
    span = 3
    result = ewma_numba(data, span)

    expected = pd.Series(data).ewm(span=span, adjust=False).mean().values

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

def test_ewma_numba_with_nans():
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    span = 3
    result = ewma_numba(data, span)

    # Pandas EWM with span, adjust=False handles NaNs by ignoring them and not advancing the decay
    # ewma_numba currently propagates NaNs (NaN * alpha + (1-alpha) * out[i-1] = NaN)
    # The subsequent values will also be NaN.
    expected = np.array([1.0, 1.5, np.nan, np.nan, np.nan])

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8, equal_nan=True)

def test_ewma_numba_empty():
    data = np.array([], dtype=np.float64)
    span = 3
    result = ewma_numba(data, span)
    expected = np.array([])
    np.testing.assert_allclose(result, expected, equal_nan=True)
