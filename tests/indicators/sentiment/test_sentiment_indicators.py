import numpy as np
import pytest

from src.indicators.sentiment.sentiment_indicators import fear_and_greed_index_numba


@pytest.fixture
def ohlcv_data():
    """200-bar realistic OHLCV data with slight uptrend + noise."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.normal(0.05, 1.0, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    volume = np.random.uniform(1e6, 5e6, n)
    return {"close": close, "high": high, "low": low, "volume": volume}


@pytest.fixture
def fgi_defaults():
    """Default parameters matching the indicator_categories wrapper."""
    return dict(
        rsi_length=14,
        macd_fast_length=12,
        macd_slow_length=26,
        macd_signal_length=9,
        mfi_length=14,
        window_size=60,
    )


class TestFearAndGreedIndex:
    def test_output_shape(self, ohlcv_data, fgi_defaults):
        result = fear_and_greed_index_numba(
            ohlcv_data["close"], ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["volume"],
            **fgi_defaults
        )
        assert result.shape == ohlcv_data["close"].shape

    def test_values_in_range(self, ohlcv_data, fgi_defaults):
        """FGI is normalised to [0, 100] after nan_to_num fills with 50."""
        result = fear_and_greed_index_numba(
            ohlcv_data["close"], ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["volume"],
            **fgi_defaults
        )
        assert (result >= 0).all() and (result <= 100).all(), \
            f"Values out of [0, 100]: min={result.min():.2f}, max={result.max():.2f}"

    def test_no_nan_in_output(self, ohlcv_data, fgi_defaults):
        """nan_to_num(nan=50) ensures no NaNs remain in the result."""
        result = fear_and_greed_index_numba(
            ohlcv_data["close"], ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["volume"],
            **fgi_defaults
        )
        assert not np.isnan(result).any(), "Output contains NaN values"

    def test_flat_data_returns_valid(self):
        """Flat close/volume (all same price) should not raise and remain in range."""
        n = 200
        close = np.full(n, 100.0, dtype=np.float64)
        high = close + 0.5
        low = close - 0.5
        volume = np.full(n, 1e6, dtype=np.float64)

        result = fear_and_greed_index_numba(
            close, high, low, volume,
            rsi_length=14, macd_fast_length=12, macd_slow_length=26,
            macd_signal_length=9, mfi_length=14, window_size=60
        )
        assert result.shape == (n,)
        assert not np.isnan(result).any()
        assert (result >= 0).all() and (result <= 100).all()

    def test_strong_uptrend_leans_greed(self):
        """Consistent price rises should push FGI toward greed (>50) over time."""
        n = 200
        close = np.linspace(50.0, 150.0, n)
        high = close + 1.0
        low = close - 1.0
        volume = np.full(n, 1e6, dtype=np.float64)

        result = fear_and_greed_index_numba(
            close, high, low, volume,
            rsi_length=14, macd_fast_length=12, macd_slow_length=26,
            macd_signal_length=9, mfi_length=14, window_size=60
        )
        # Average of the non-fallback-filled portion should lean toward greed
        valid = result[result != 50.0]
        if len(valid) > 0:
            assert np.mean(valid) > 50.0, \
                f"Expected greed-leaning FGI for uptrend, got mean={np.mean(valid):.2f}"

    def test_small_window_size(self):
        """Smaller window_size should still produce valid output."""
        n = 100
        np.random.seed(0)
        close = 100.0 + np.cumsum(np.random.normal(0, 1, n))
        high = close + 1.0
        low = close - 1.0
        volume = np.full(n, 1e6, dtype=np.float64)

        result = fear_and_greed_index_numba(
            close, high, low, volume,
            rsi_length=7, macd_fast_length=6, macd_slow_length=13,
            macd_signal_length=5, mfi_length=7, window_size=30
        )
        assert result.shape == (n,)
        assert (result >= 0).all() and (result <= 100).all()

    def test_output_dtype_is_float(self, ohlcv_data, fgi_defaults):
        result = fear_and_greed_index_numba(
            ohlcv_data["close"], ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["volume"],
            **fgi_defaults
        )
        assert result.dtype == np.float64 or np.issubdtype(result.dtype, np.floating)
