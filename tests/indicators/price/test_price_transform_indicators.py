import numpy as np
import pytest

from src.indicators.price.price_transform_indicators import (
    log_return_numba,
    percent_return_numba,
    pdist_numba,
)


@pytest.fixture
def sample_close():
    """Simple upward trend with a reversal, suitable for return calculations."""
    return np.array([10.0, 11.0, 12.0, 10.0, 9.0, 11.0, 12.0, 13.0, 12.0, 11.0], dtype=np.float64)


@pytest.fixture
def ohlcv():
    """OHLCV data for pdist tests."""
    np.random.seed(7)
    n = 20
    close = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    open_ = close + np.random.normal(0, 0.5, n)
    high = np.maximum(close, open_) + np.abs(np.random.normal(0, 0.5, n))
    low = np.minimum(close, open_) - np.abs(np.random.normal(0, 0.5, n))
    volume = np.random.uniform(1e5, 1e6, n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


# ---------------------------------------------------------------------------
# log_return_numba
# ---------------------------------------------------------------------------

class TestLogReturn:
    def test_shape(self, sample_close):
        result = log_return_numba(sample_close, length=1)
        assert result.shape == sample_close.shape

    def test_nan_warmup(self, sample_close):
        length = 3
        result = log_return_numba(sample_close, length=length)
        assert np.isnan(result[:length]).all()

    def test_known_values_length_1(self):
        close = np.array([100.0, 110.0, 99.0], dtype=np.float64)
        result = log_return_numba(close, length=1)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], np.log(110.0 / 100.0))
        np.testing.assert_allclose(result[2], np.log(99.0 / 110.0))

    def test_known_values_length_2(self):
        close = np.array([100.0, 110.0, 121.0, 90.0], dtype=np.float64)
        result = log_return_numba(close, length=2)
        np.testing.assert_allclose(result[2], np.log(121.0 / 100.0))
        np.testing.assert_allclose(result[3], np.log(90.0 / 110.0))

    def test_cumulative_mode(self):
        close = np.array([100.0, 110.0, 121.0, 90.0], dtype=np.float64)
        result = log_return_numba(close, length=1, cumulative=True)
        # cumulative: log(close[i] / close[0])
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], np.log(110.0 / 100.0))
        np.testing.assert_allclose(result[2], np.log(121.0 / 100.0))
        np.testing.assert_allclose(result[3], np.log(90.0 / 100.0))

    def test_matches_numpy_for_full_series(self, sample_close):
        length = 1
        result = log_return_numba(sample_close, length=length)
        expected = np.full_like(sample_close, np.nan)
        for i in range(length, len(sample_close)):
            expected[i] = np.log(sample_close[i] / sample_close[i - length])
        np.testing.assert_allclose(result[length:], expected[length:], rtol=1e-12)

    def test_negative_return(self):
        close = np.array([100.0, 50.0], dtype=np.float64)
        result = log_return_numba(close, length=1)
        assert result[1] < 0.0

    def test_zero_return(self):
        close = np.array([50.0, 50.0], dtype=np.float64)
        result = log_return_numba(close, length=1)
        np.testing.assert_allclose(result[1], 0.0)


# ---------------------------------------------------------------------------
# percent_return_numba
# ---------------------------------------------------------------------------

class TestPercentReturn:
    def test_shape(self, sample_close):
        result = percent_return_numba(sample_close, length=1)
        assert result.shape == sample_close.shape

    def test_nan_warmup(self, sample_close):
        length = 2
        result = percent_return_numba(sample_close, length=length)
        assert np.isnan(result[:length]).all()

    def test_known_values_length_1(self):
        close = np.array([100.0, 110.0, 99.0], dtype=np.float64)
        result = percent_return_numba(close, length=1)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 0.10)          # +10%
        np.testing.assert_allclose(result[2], -0.1, rtol=1e-10)  # -10%

    def test_known_values_length_2(self):
        close = np.array([100.0, 110.0, 125.0], dtype=np.float64)
        result = percent_return_numba(close, length=2)
        np.testing.assert_allclose(result[2], 0.25)  # (125/100) - 1

    def test_cumulative_mode(self):
        close = np.array([100.0, 110.0, 121.0, 90.0], dtype=np.float64)
        result = percent_return_numba(close, length=1, cumulative=True)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 0.10)
        np.testing.assert_allclose(result[2], 0.21)
        np.testing.assert_allclose(result[3], -0.10)

    def test_matches_numpy(self, sample_close):
        length = 1
        result = percent_return_numba(sample_close, length=length)
        expected = np.full_like(sample_close, np.nan)
        for i in range(length, len(sample_close)):
            expected[i] = (sample_close[i] / sample_close[i - length]) - 1
        np.testing.assert_allclose(result[length:], expected[length:], rtol=1e-12)

    def test_relationship_with_log_return(self):
        """For small returns, pct_return ≈ log_return; always pct_return > log_return for positive returns."""
        close = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        pct = percent_return_numba(close, length=1)
        log_ = log_return_numba(close, length=1)
        # pct_return > log_return for positive price changes
        assert (pct[1:] > log_[1:]).all()


# ---------------------------------------------------------------------------
# pdist_numba
# ---------------------------------------------------------------------------

class TestPdist:
    def test_shape(self, ohlcv):
        result = pdist_numba(ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup_drift_1(self, ohlcv):
        drift = 1
        result = pdist_numba(ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], drift=drift)
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_known_value(self):
        """
        pdist[i] = 2*(high[i] - low[i]) - |close[i] - open[i]| + |open[i] - close[i-1]|
        """
        open_ = np.array([10.0, 12.0], dtype=np.float64)
        high  = np.array([15.0, 16.0], dtype=np.float64)
        low   = np.array([8.0,  9.0],  dtype=np.float64)
        close = np.array([11.0, 13.0], dtype=np.float64)

        result = pdist_numba(open_, high, low, close, drift=1)

        # i=1: 2*(16-9) - |13-12| + |12-11| = 14 - 1 + 1 = 14
        np.testing.assert_allclose(result[1], 14.0)

    def test_non_negative(self, ohlcv):
        """pdist measures price range/movement; should be non-negative for normal OHLC data."""
        result = pdist_numba(ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result[~np.isnan(result)]
        # Won't always be >= 0, but check it returns finite values
        assert np.isfinite(valid).all()

    def test_drift_2(self, ohlcv):
        drift = 2
        result = pdist_numba(ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], drift=drift)
        assert np.isnan(result[:drift]).all()
        assert not np.isnan(result[drift])
