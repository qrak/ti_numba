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

    # test default (rma)
    atr_rma = atr_numba(high, low, close, length=length, mamode='rma', percent=False)
    assert len(atr_rma) == len(close)
    # the exact number of nans depends on the mode, but from `length` onwards it should be valid
    assert not np.isnan(atr_rma[length:]).any()

def test_atr_numba_modes():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0, 21.0, 22.0, 23.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0, 17.0, 19.0, 20.0, 21.0, 22.0])
    length = 5

    modes = ['rma', 'ema', 'sma', 'wma']
    for mode in modes:
        atr = atr_numba(high, low, close, length=length, mamode=mode, percent=False)
        assert len(atr) == len(close)
        assert not np.isnan(atr[length:]).any()

def test_atr_numba_percent():
    high = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0])
    length = 3

    atr_percent = atr_numba(high, low, close, length=length, percent=True)
    # The first 'length' elements might be nan, check the valid ones
    assert not np.isnan(atr_percent[length:]).any()

def test_atr_numba_nan_inputs():
    high = np.array([10.0, 12.0, np.nan, 14.0, 16.0])
    low = np.array([8.0, 9.0, 11.0, 10.0, 12.0])
    close = np.array([9.0, 11.0, 14.0, 12.0, 15.0])

    atr = atr_numba(high, low, close, length=3)
    assert np.isinf(atr).all()

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
    assert not np.isnan(long_exit[length - 1:]).any()
    assert not np.isnan(short_exit[length - 1:]).any()

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
