import numpy as np
from src.indicators.volatility.volatility_indicators import (
    atr_numba,
    bollinger_bands_numba,
    chandelier_exit_numba,
    ebsw_numba,
    vhf_numba
)

def test_atr_numba_basic():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0, 14.0, 15.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0, 17.0, 19.0])
    length = 3

    # Calculate ATR manually for comparison (RMA default)
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan
    expected_atr[length] = np.mean(tr[1:length+1]) # Initial RMA seed is SMA of 'length' items
    for i in range(length + 1, len(close)):
        expected_atr[i] = (expected_atr[i - 1] * (length - 1) + tr[i]) / length

    # test default (rma)
    atr_rma = atr_numba(high, low, close, length=length, mamode='rma', percent=False)
    assert len(atr_rma) == len(close)
    np.testing.assert_allclose(atr_rma[length:], expected_atr[length:], equal_nan=True)

def test_atr_numba_modes():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0, 21.0, 22.0, 23.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0, 17.0, 19.0, 20.0, 21.0, 22.0])
    length = 5

    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Test EMA
    expected_ema = np.empty(len(close))
    expected_ema[:length] = np.nan
    expected_ema[length] = np.mean(tr[1:length+1])
    alpha = 2 / (length + 1)
    for i in range(length + 1, len(close)):
        expected_ema[i] = (1 - alpha) * expected_ema[i - 1] + alpha * tr[i]
    atr_ema = atr_numba(high, low, close, length=length, mamode='ema', percent=False)
    np.testing.assert_allclose(atr_ema[length:], expected_ema[length:], equal_nan=True)

    # Test SMA against manual sliding window (pandas-equivalent)
    expected_sma = np.empty(len(close))
    expected_sma[:length] = np.nan
    for i in range(length, len(close)):
        expected_sma[i] = np.mean(tr[i - length + 1 : i + 1])
    atr_sma = atr_numba(high, low, close, length=length, mamode='sma', percent=False)
    np.testing.assert_allclose(atr_sma[length:], expected_sma[length:], equal_nan=True)

    # Test WMA
    expected_wma = np.empty(len(close))
    expected_wma[:length] = np.nan
    weights = np.arange(1, length + 1).astype(np.float64)
    weight_sum = np.sum(weights)
    for i in range(length, len(close)):
        expected_wma[i] = np.dot(tr[i - length + 1:i + 1], weights) / weight_sum
    atr_wma = atr_numba(high, low, close, length=length, mamode='wma', percent=False)
    np.testing.assert_allclose(atr_wma[length:], expected_wma[length:], equal_nan=True)

def test_atr_numba_percent():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0, 14.0, 15.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0, 17.0, 19.0])
    length = 3

    # Calculate ATR manually for comparison
    tr = np.empty(len(close))
    tr[0] = 0
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    expected_atr = np.empty(len(close))
    expected_atr[:length] = np.nan
    expected_atr[length] = np.mean(tr[1:length+1])
    for i in range(length + 1, len(close)):
        expected_atr[i] = (expected_atr[i - 1] * (length - 1) + tr[i]) / length

    expected_atr[length:] *= 100 / close[length:]

    atr_percent = atr_numba(high, low, close, length=length, mamode='rma', percent=True)
    np.testing.assert_allclose(atr_percent[length:], expected_atr[length:], equal_nan=True)

def test_atr_numba_nan_inputs():
    high = np.array([10.0, 12.0, np.nan, 14.0, 16.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0])

    atr = atr_numba(high, low, close, length=3)
    assert np.isnan(atr).all()

    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
    low = np.array([8.0, 9.0, np.nan, 10.0, 12.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0])
    atr = atr_numba(high, low, close, length=3)
    assert np.isnan(atr).all()

    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0])
    close = np.array([9.0, 11.0, np.nan, 12.0, 15.0])
    atr = atr_numba(high, low, close, length=3)
    assert np.isnan(atr).all()

def test_bollinger_bands_numba():
    # Simple linear trend
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    length = 3
    num_std_dev = 2.0

    upper, middle, lower = bollinger_bands_numba(close, length, num_std_dev)

    assert len(upper) == len(close)
    assert len(middle) == len(close)
    assert len(lower) == len(close)

    assert np.isnan(upper[:length - 1]).all()
    assert np.isnan(middle[:length - 1]).all()
    assert np.isnan(lower[:length - 1]).all()

    # Calculate expected for index 2: window = [10, 11, 12]
    # mean = 11
    # std = sqrt(((10-11)^2 + (11-11)^2 + (12-11)^2) / 3) = sqrt(2/3) ~ 0.816496
    expected_mean = 11.0
    expected_std = np.sqrt(2.0 / 3.0)

    assert np.isclose(middle[2], expected_mean)
    assert np.isclose(upper[2], expected_mean + num_std_dev * expected_std)
    assert np.isclose(lower[2], expected_mean - num_std_dev * expected_std)

def test_chandelier_exit_numba():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0, 14.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0, 17.0])
    length = 3
    multiplier = 2.0

    long_exit, short_exit = chandelier_exit_numba(high, low, close, length, multiplier)

    assert len(long_exit) == len(close)
    assert len(short_exit) == len(close)
    assert np.all(long_exit[:length - 1] == 0) # The first `length-1` are untouched
    assert np.all(short_exit[:length - 1] == 0)
    # The rest shouldn't be zero typically unless calculated as such
    assert not np.isnan(long_exit[length:]).any()
    assert not np.isnan(short_exit[length:]).any()

def test_ebsw_numba():
    close = np.array([10.0 + i for i in range(50)]) # Trend
    length = 40

    ebsw = ebsw_numba(close, length=length)

    assert len(ebsw) == len(close)
    assert np.isnan(ebsw[:length]).all()
    assert not np.isnan(ebsw[length:]).any()

def test_vhf_numba():
    close = np.array([10.0, 12.0, 11.0, 14.0, 13.0, 16.0, 15.0])
    length = 3
    drift = 1

    vhf = vhf_numba(close, length=length, drift=drift)

    assert len(vhf) == len(close)
    assert np.isnan(vhf[:length - 1 + drift]).all()
    assert not np.isnan(vhf[length - 1 + drift:]).any()

    # Division by zero test
    flat_close = np.array([10.0]*10)
    vhf_flat = vhf_numba(flat_close, length=3, drift=1)
    # The valid part should be 0, no runtime warning or division by zero error should happen
    assert (vhf_flat[3:] == 0.0).all()

    # Short length test
    short_close = np.array([10.0, 12.0])
    vhf_short = vhf_numba(short_close, length=5)
    assert np.isnan(vhf_short).all()
