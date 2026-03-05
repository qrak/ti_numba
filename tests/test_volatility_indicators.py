import numpy as np
import pytest

from src.indicators.volatility.volatility_indicators import atr_numba

def test_atr_numba_rma():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
    length = 5

    # Calculate ATR manually for comparison
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan
    expected_atr[length - 1] = np.mean(tr[1:length])
    for i in range(length, len(close)):
        expected_atr[i] = (expected_atr[i - 1] * (length - 1) + tr[i]) / length

    atr = atr_numba(high, low, close, length=length, mamode='rma', percent=False)

    np.testing.assert_array_almost_equal(atr[length-1:], expected_atr[length-1:])
    assert np.isnan(atr[:length-1]).all()

def test_atr_numba_ema():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
    length = 5

    # Calculate ATR manually for comparison
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan
    expected_atr[length - 1] = np.mean(tr[1:length])
    alpha = 2 / (length + 1)
    for i in range(length, len(close)):
        expected_atr[i] = (1 - alpha) * expected_atr[i - 1] + alpha * tr[i]

    atr = atr_numba(high, low, close, length=length, mamode='ema', percent=False)

    np.testing.assert_array_almost_equal(atr[length-1:], expected_atr[length-1:])
    assert np.isnan(atr[:length-1]).all()

def test_atr_numba_sma():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
    length = 5

    # Calculate ATR manually for comparison
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan

    sum_tr = np.sum(tr[1:length])
    for i in range(length, len(close)):
        expected_atr[i] = sum_tr / length
        sum_tr += tr[i] - tr[i - length + 1]

    atr = atr_numba(high, low, close, length=length, mamode='sma', percent=False)

    np.testing.assert_array_almost_equal(atr[length:], expected_atr[length:])
    assert np.isnan(atr[:length]).all()

def test_atr_numba_wma():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
    length = 5

    # Calculate ATR manually for comparison
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan

    weights = np.arange(1, length + 1).astype(np.float64)
    weight_sum = np.sum(weights)
    for i in range(length, len(close)):
        expected_atr[i] = np.dot(tr[i - length + 1:i + 1], weights) / weight_sum

    atr = atr_numba(high, low, close, length=length, mamode='wma', percent=False)

    np.testing.assert_array_almost_equal(atr[length:], expected_atr[length:])
    assert np.isnan(atr[:length]).all()

def test_atr_numba_percent():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
    length = 5

    # Calculate ATR manually for comparison
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan
    expected_atr[length - 1] = np.mean(tr[1:length])
    for i in range(length, len(close)):
        expected_atr[i] = (expected_atr[i - 1] * (length - 1) + tr[i]) / length

    expected_atr[length:] *= 100 / close[length:]

    atr = atr_numba(high, low, close, length=length, mamode='rma', percent=True)

    np.testing.assert_array_almost_equal(atr[length-1:], expected_atr[length-1:])

def test_atr_numba_nan_inputs():
    high = np.array([10.0, 12.0, np.nan, 14.0, 16.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0])
    length = 3

    atr = atr_numba(high, low, close, length=length)
    assert np.isinf(atr).all()

    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
    low = np.array([8.0, 9.0, np.nan, 12.0, 14.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0])
    atr = atr_numba(high, low, close, length=length)
    assert np.isinf(atr).all()

    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0])
    close = np.array([9.0, 11.0, np.nan, 13.0, 15.0])
    atr = atr_numba(high, low, close, length=length)
    assert np.isinf(atr).all()

def test_atr_numba_default_mamode_and_percent():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    low = np.array([8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    close = np.array([9.0, 11.0, 14.0, 13.0, 15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])

    # Calculate ATR manually for comparison (mamode='rma', percent=False)
    length = 14
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan
    expected_atr[length - 1] = np.mean(tr[1:length])
    for i in range(length, len(close)):
        expected_atr[i] = (expected_atr[i - 1] * (length - 1) + tr[i]) / length

    atr = atr_numba(high, low, close, length)

    np.testing.assert_array_almost_equal(atr[length-1:], expected_atr[length-1:])
    assert np.isnan(atr[:length-1]).all()
