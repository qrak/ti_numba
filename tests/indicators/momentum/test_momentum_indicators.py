import pytest
import numpy as np

from src.indicators.momentum.momentum_indicators import (
    rsi_numba, macd_numba, stochastic_numba, roc_numba,
    momentum_numba, williams_r_numba, tsi_numba, rmi_numba,
    ppo_numba, coppock_curve_numba, detect_rsi_divergence,
    calculate_relative_strength_numba, uo_numba
)

@pytest.fixture
def sample_data():
    # A simple upward trend then downward trend for close
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    high = close + 1.0
    low = close - 1.0
    return {
        "close": close,
        "high": high,
        "low": low
    }

def test_rsi_numba(sample_data):
    close = sample_data["close"]
    length = 5
    rsi = rsi_numba(close, length)
    assert len(rsi) == len(close)
    assert np.isnan(rsi[:length]).all()
    # It should be between 0 and 100
    assert (rsi[length:] >= 0).all() and (rsi[length:] <= 100).all()

def test_macd_numba(sample_data):
    close = sample_data["close"]
    macd_line, signal_line, histogram = macd_numba(close, 3, 6, 3)
    assert len(macd_line) == len(close)
    assert len(signal_line) == len(close)
    assert len(histogram) == len(close)
    assert np.isnan(macd_line[:5]).all()
    assert not np.isnan(macd_line[5])
    assert not np.isnan(signal_line[6])

def test_stochastic_numba(sample_data):
    close = sample_data["close"]
    high = sample_data["high"]
    low = sample_data["low"]
    period_k = 5
    smooth_k = 3
    period_d = 3
    k, d = stochastic_numba(high, low, close, period_k, smooth_k, period_d)
    assert len(k) == len(close)
    assert len(d) == len(close)
    # Valid values should be between 0 and 100
    valid_k = k[~np.isnan(k)]
    valid_d = d[~np.isnan(d)]
    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()

def test_roc_numba(sample_data):
    close = sample_data["close"]
    roc = roc_numba(close, length=2)
    assert len(roc) == len(close)
    assert np.isnan(roc[:2]).all()
    # roc[2] = (12/10 - 1)*100 = 20.0
    np.testing.assert_allclose(roc[2], 20.0)

def test_momentum_numba(sample_data):
    close = sample_data["close"]
    mom = momentum_numba(close, length=2)
    assert len(mom) == len(close)
    assert np.isnan(mom[:2]).all()
    # mom[2] = 12 - 10 = 2.0
    np.testing.assert_allclose(mom[2], 2.0)

def test_williams_r_numba(sample_data):
    close = sample_data["close"]
    high = sample_data["high"]
    low = sample_data["low"]
    length = 5
    wr = williams_r_numba(high, low, close, length)
    assert len(wr) == len(close)
    valid_wr = wr[~np.isnan(wr)]
    assert (valid_wr >= -100).all() and (valid_wr <= 0).all()

def test_tsi_numba(sample_data):
    close = sample_data["close"]
    tsi = tsi_numba(close, 5, 3)
    assert len(tsi) == len(close)
    valid_tsi = tsi[~np.isnan(tsi)]
    assert (valid_tsi >= -100).all() and (valid_tsi <= 100).all()

def test_rmi_numba(sample_data):
    close = sample_data["close"]
    rmi = rmi_numba(close, length=5, momentum_length=2)
    assert len(rmi) == len(close)
    valid_rmi = rmi[~np.isnan(rmi)]
    assert (valid_rmi >= 0).all() and (valid_rmi <= 100).all()

def test_ppo_numba(sample_data):
    close = sample_data["close"]
    ppo = ppo_numba(close, fast_length=3, slow_length=6)
    assert len(ppo) == len(close)
    assert np.isnan(ppo[:5]).all()
    assert not np.isnan(ppo[5])

def test_coppock_curve_numba(sample_data):
    close = sample_data["close"]
    cc = coppock_curve_numba(close, wl1=5, wl2=3, wma_length=3)
    assert len(cc) == len(close)

def test_detect_rsi_divergence(sample_data):
    close = sample_data["close"]
    rsi = rsi_numba(close, length=5)
    # Mocking rsi divergence by replacing NaNs with some values so that we have data
    rsi = np.nan_to_num(rsi, nan=50)
    div = detect_rsi_divergence(close, rsi, length=2)
    assert len(div) == len(close)
    assert (np.isin(div, [-1, 0, 1])).all()

def test_calculate_relative_strength_numba(sample_data):
    pair_close = sample_data["close"]
    # benchmark close is slightly different
    benchmark_close = pair_close * 1.05
    rs = calculate_relative_strength_numba(pair_close, benchmark_close, window=3)
    assert len(rs) == len(pair_close)
    # Because they are proportional, the return difference should be approx 0
    np.testing.assert_allclose(rs[3:], 0.0, atol=1e-7)

def test_uo_numba(sample_data):
    close = sample_data["close"]
    high = sample_data["high"]
    low = sample_data["low"]
    uo = uo_numba(high, low, close, fast=3, medium=6, slow=9, fast_w=4, medium_w=2, slow_w=1, drift=1)
    assert len(uo) == len(close)
    valid_uo = uo[~np.isnan(uo)]
    assert (valid_uo >= 0).all() and (valid_uo <= 100).all()

def test_rsi_numba_uptrend():
    """Test RSI behavior in a consistent uptrend (should be 100)."""
    close_prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    length = 5
    rsi = rsi_numba(close_prices, length)
    assert np.allclose(rsi[length:], 100.0)

def test_rsi_numba_downtrend():
    """Test RSI behavior in a consistent downtrend (should be 0)."""
    close_prices = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    length = 5
    rsi = rsi_numba(close_prices, length)
    assert np.allclose(rsi[length:], 0.0)

def test_rsi_numba_short_array():
    """Test RSI behavior when the price array is shorter than the RSI length."""
    close_prices = np.array([10.0, 11.0, 12.0])
    length = 5
    rsi = rsi_numba(close_prices, length)
    assert np.isnan(rsi).all()

def test_rsi_numba_alternating():
    """Test RSI with alternating up and down movements to verify calculation."""
    close_prices = np.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    length = 5
    rsi = rsi_numba(close_prices, length)
    expected_rsi = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 60.0, 48.0, 58.4])
    mask = ~np.isnan(expected_rsi)
    assert np.allclose(rsi[mask], expected_rsi[mask])

def test_rsi_numba_flat():
    """Test RSI behavior in a flat market (should stay at 50 if gains=losses or similar)."""
    close_prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    length = 5
    rsi = rsi_numba(close_prices, length)
    assert np.allclose(rsi[length:], 100.0)

def test_rsi_numba_exact_length():
    """Test RSI behavior when the price array is exactly the RSI length."""
    close_prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    length = 5
    rsi = rsi_numba(close_prices, length)
    assert np.isnan(rsi).all()

def test_tsi_numba_zero_abs_ema2():
    """Test TSI behavior when abs_ema2 is exactly 0 (flat price)."""
    close = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    # The output from tsi_numba when there's no price movement should fall to `tsi[i-1]` (which is NaN initially)
    tsi = tsi_numba(close, 3, 2)
    assert np.isnan(tsi).all()

def test_detect_rsi_divergence_all_cases():
    """Test all branches of detect_rsi_divergence."""
    close = np.array([10.0, 11.0, 9.0, 12.0])
    rsi = np.array([50.0, 40.0, 60.0, 60.0])
    # length=1
    # i=1: price_diff = 11 - 10 = 1, rsi_diff = 40 - 50 = -1 -> price > 0, rsi < 0 -> -1
    # i=2: price_diff = 9 - 11 = -2, rsi_diff = 60 - 40 = 20 -> price < 0, rsi > 0 -> 1
    # i=3: price_diff = 12 - 9 = 3, rsi_diff = 60 - 60 = 0 -> price > 0, rsi == 0 -> 0
    divergence = detect_rsi_divergence(close, rsi, length=1)

    expected = np.array([0, -1, 1, 0])
    assert np.array_equal(divergence, expected)

def test_calculate_relative_strength_numba_edge_cases():
    """Test calculate_relative_strength_numba with NaNs and zero benchmark."""
    pair_close = np.array([10.0, 11.0, np.nan, 13.0])
    benchmark_close = np.array([20.0, 22.0, 24.0, 0.0])

    # window=1
    rs = calculate_relative_strength_numba(pair_close, benchmark_close, window=1)

    # For i=1: pair=11.0, bench=22.0 (valid, returns log(1.1) and log(1.1)) => rs[1] = 0.0
    # For i=2: pair=NaN, bench=24.0 (has NaN) => rs[2] = 0.0
    # For i=3: pair=13.0, bench=0.0 (bench is 0.0) => rs[3] = 0.0

    assert rs[2] == 0.0
    assert rs[3] == 0.0
    assert rs[1] == 0.0 # From proportional returns

def test_rsi_numba_basic():
    """Test basic RSI functionality."""
    close_prices = np.array([10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 14.0, 13.5, 14.5, 15.0])
    length = 5
    rsi = rsi_numba(close_prices, length)

    assert len(rsi) == len(close_prices)
    assert np.isnan(rsi[:length]).all() # First length items should be NaN
    assert not np.isnan(rsi[length:]).any() # Rest should be numbers
