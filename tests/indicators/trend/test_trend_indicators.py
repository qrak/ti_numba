import numpy as np
import pytest

from src.indicators.trend.trend_indicators import (
    adx_numba,
    supertrend_numba,
    ichimoku_cloud_numba,
    parabolic_sar_numba,
    trix_numba,
    vortex_indicator_numba,
    pfe_numba
)

@pytest.fixture
def ohlcv_data():
    """Returns sample OHLC data for testing."""
    np.random.seed(42)
    n = 100

    # Generate some random walk data for testing
    close_prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    high_prices = close_prices * (1 + np.random.uniform(0, 0.02, n))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.02, n))

    return {
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }

def test_adx_numba(ohlcv_data):
    high, low, close = ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
    length = 14

    # Pre-compile the numba function by calling it once and then check the output
    adx, pdi, ndi = adx_numba(high, low, close, length)

    # Check shapes
    assert adx.shape == close.shape
    assert pdi.shape == close.shape
    assert ndi.shape == close.shape

    # Check initial values are NaN
    assert np.isnan(adx[:(length * 2 - 2)]).all()

    # Check bounds (ADX, +DI, -DI should be between 0 and 100)
    valid_adx = adx[~np.isnan(adx)]
    valid_pdi = pdi[~np.isnan(pdi)]
    valid_ndi = ndi[~np.isnan(ndi)]

    if len(valid_adx) > 0:
        assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
    if len(valid_pdi) > 0:
        assert (valid_pdi >= 0).all() and (valid_pdi <= 100).all()
    if len(valid_ndi) > 0:
        assert (valid_ndi >= 0).all() and (valid_ndi <= 100).all()

def test_supertrend_numba(ohlcv_data):
    high, low, close = ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
    length = 10
    multiplier = 3.0

    trend, direction = supertrend_numba(high, low, close, length, multiplier)

    # Check shapes
    assert trend.shape == close.shape
    assert direction.shape == close.shape

    # Check direction values are 1 or -1
    unique_directions = np.unique(direction)
    assert set(unique_directions).issubset({1, -1})

def test_ichimoku_cloud_numba(ohlcv_data):
    high, low = ohlcv_data['high'], ohlcv_data['low']
    conversion_length = 9
    base_length = 26
    lagging_span2_length = 52
    displacement = 26

    conversion, base, span_a, span_b = ichimoku_cloud_numba(
        high, low, conversion_length, base_length, lagging_span2_length, displacement
    )

    assert conversion.shape == high.shape
    assert base.shape == high.shape
    assert span_a.shape == high.shape
    assert span_b.shape == high.shape

    # Check early values are NaN
    assert np.isnan(conversion[:conversion_length - 1]).all()
    assert np.isnan(base[:base_length - 1]).all()

def test_parabolic_sar_numba(ohlcv_data):
    high, low = ohlcv_data['high'], ohlcv_data['low']
    sar = parabolic_sar_numba(high, low)

    assert sar.shape == high.shape
    assert not np.isnan(sar).all()

def test_trix_numba(ohlcv_data):
    close = ohlcv_data['close']
    trix = trix_numba(close, length=5)  # Use small length to avoid all NaNs

    assert trix.shape == close.shape

def test_vortex_indicator_numba(ohlcv_data):
    high, low, close = ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
    vi_plus, vi_minus = vortex_indicator_numba(high, low, close, length=14)

    assert vi_plus.shape == close.shape
    assert vi_minus.shape == close.shape
    assert np.isnan(vi_plus[:14]).all()

def test_pfe_numba(ohlcv_data):
    close = ohlcv_data['close']
    n = 10
    m = 5
    pfe = pfe_numba(close, n, m)

    assert pfe.shape == close.shape
    assert np.isnan(pfe[:n - 1]).all()
