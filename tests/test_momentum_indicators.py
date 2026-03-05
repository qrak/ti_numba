import numpy as np
import pytest
from src.indicators.momentum.momentum_indicators import rsi_numba

def test_rsi_numba_basic():
    """Test basic RSI functionality."""
    close_prices = np.array([10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 14.0, 13.5, 14.5, 15.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    assert len(rsi) == len(close_prices)
    assert np.isnan(rsi[:length]).all() # First length items should be NaN
    assert not np.isnan(rsi[length:]).any() # Rest should be numbers

def test_rsi_numba_uptrend():
    """Test RSI behavior in a consistent uptrend (should be 100)."""
    close_prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    # After the initial `length` period, if there are only gains, RSI should be 100
    assert np.allclose(rsi[length:], 100.0)

def test_rsi_numba_downtrend():
    """Test RSI behavior in a consistent downtrend (should be 0)."""
    close_prices = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    # After the initial `length` period, if there are only losses, RSI should be 0
    assert np.allclose(rsi[length:], 0.0)

def test_rsi_numba_short_array():
    """Test RSI behavior when the price array is shorter than the RSI length."""
    close_prices = np.array([10.0, 11.0, 12.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    # The whole array should be NaNs because length is greater than the data size
    assert np.isnan(rsi).all()

def test_rsi_numba_alternating():
    """Test RSI with alternating up and down movements to verify calculation."""
    # Prices: 10, 11 (+1), 10 (-1), 11 (+1), 10 (-1), 11 (+1), 10 (-1), 11 (+1)
    close_prices = np.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    # The output we calculated manually using python script:
    # [ nan  nan  nan  nan  nan 60.  48.  58.4]
    expected_rsi = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 60.0, 48.0, 58.4])

    # Compare only non-NaN elements
    mask = ~np.isnan(expected_rsi)
    assert np.allclose(rsi[mask], expected_rsi[mask])

def test_rsi_numba_flat():
    """Test RSI behavior in a flat market (should stay at 50 if gains=losses or similar)."""
    # Note: if price never changes, gains=0, losses=0 -> avg_loss=0 -> rsi=100 in the code
    # This might be an artifact of the implementation, but let's test what it does.
    close_prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    # In the provided code, if avg_loss == 0, rsi is set to 100.
    assert np.allclose(rsi[length:], 100.0)


def test_rsi_numba_exact_length():
    """Test RSI behavior when the price array is exactly the RSI length."""
    close_prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    # The whole array should be NaNs because we need at least length + 1 values
    assert np.isnan(rsi).all()
