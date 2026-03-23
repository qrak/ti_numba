import pytest
import numpy as np

from src.indicators.volume.volume_indicators import (
    obv_numba,
    mfi_numba,
    pvt_numba,
    chaikin_money_flow_numba,
    ad_line_numba,
    force_index_numba,
    eom_numba,
    volume_profile_numba,
    rolling_vwap_numba,
    twap_numba,
    cci_numba,
    average_quote_volume_numba,
)


@pytest.fixture
def ohlcv():
    """50-bar realistic OHLCV fixture for most volume tests."""
    np.random.seed(99)
    n = 50
    close = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    volume = np.random.uniform(1e5, 1e6, n)
    return {"close": close, "high": high, "low": low, "volume": volume, "n": n}


# ---------------------------------------------------------------------------
# obv_numba (original tests preserved)
# ---------------------------------------------------------------------------

def test_obv_numba_basic():
    close = np.array([10.0, 11.0, 10.0, 12.0, 12.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0, 100.0])
    length = 2
    # obv[1] = 1 * 200 = 200
    # i=2: 10 < 11 → 200 - 150 = 50
    # i=3: 12 > 10 → 50 + 300 = 350
    # i=4: 12 == 12 → 350
    expected = np.array([np.nan, 200.0, 50.0, 350.0, 350.0])
    result = obv_numba(close, volume, length)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_obv_numba_custom_initial():
    close = np.array([10.0, 11.0, 10.0, 12.0, 12.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0, 100.0])
    expected = np.array([np.nan, -200.0, -350.0, -50.0, -50.0])
    result = obv_numba(close, volume, length=2, initial=-1)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_obv_numba_length_equals_array_length():
    close = np.array([10.0, 11.0])
    volume = np.array([100.0, 200.0])
    expected = np.array([np.nan, 200.0])
    result = obv_numba(close, volume, length=2)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_obv_numba_all_same_close():
    close = np.array([10.0, 10.0, 10.0, 10.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0])
    expected = np.array([100.0, 100.0, 100.0, 100.0])
    result = obv_numba(close, volume, length=1)
    np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# mfi_numba
# ---------------------------------------------------------------------------

class TestMFI:
    def test_shape(self, ohlcv):
        result = mfi_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=14)
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 14
        result = mfi_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=length)
        assert np.isnan(result[:length]).all()

    def test_values_in_range(self, ohlcv):
        """MFI must lie in [0, 100]."""
        result = mfi_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=14)
        valid = result[~np.isnan(result)]
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_extreme_buying_gives_100(self):
        """All positive money flow → MFI = 100."""
        n = 20
        close = np.linspace(100, 120, n)
        high  = close + 1.0
        low   = close - 1.0
        volume = np.full(n, 1e6)
        result = mfi_numba(high, low, close, volume, length=5)
        # After warmup, with strictly increasing typical price, nmf=0 → MFI=100
        valid = result[~np.isnan(result)]
        assert (valid == 100.0).all()


# ---------------------------------------------------------------------------
# pvt_numba
# ---------------------------------------------------------------------------

class TestPVT:
    def test_shape(self, ohlcv):
        result = pvt_numba(ohlcv["close"], ohlcv["volume"], length=1)
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 2
        result = pvt_numba(ohlcv["close"], ohlcv["volume"], length=length)
        assert np.isnan(result[:length - 1]).all()

    def test_known_value(self):
        """
        PVT is cumulative: pv += roc * volume where roc = (close[i] - close[i-drift]) / close[i-drift].
        For length=1, drift=1:
        i=0: roc = (close[0]-close[-1])/close[-1] (Numba behavior for -1 index)
        To be safe and deterministic, let's test from length > 1 or i >= drift.
        """
        close  = np.array([10.0, 10.0, 12.0, 9.0], dtype=np.float64)
        volume = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float64)
        # length=2 means pvt[0] is NaN, calculation starts at i=1
        result = pvt_numba(close, volume, length=2, drift=1)

        # i=1: roc = (10-10)/10 = 0; pv = 0 + 0*100 = 0; pvt[1] = 0
        # i=2: roc = (12-10)/10 = 0.2; pv = 0 + 0.2*200 = 40; pvt[2] = 40
        # i=3: roc = (9-12)/12 = -0.25; pv = 40 + (-0.25)*300 = 40 - 75 = -35; pvt[3] = -35

        expected = np.array([np.nan, 0.0, 40.0, -35.0])
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_constant_price(self):
        n = 10
        close = np.full(n, 50.0)
        volume = np.random.uniform(100, 1000, n)
        result = pvt_numba(close, volume, length=1)
        # roc will be 0 everywhere (except potentially i=0 if close[-1] != 50, but here it is)
        # We expect 0 after warmup
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_zero_volume(self):
        n = 10
        close = np.linspace(100, 110, n)
        volume = np.zeros(n)
        result = pvt_numba(close, volume, length=1)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_rising_market_positive_pvt(self):
        n = 20
        close = np.linspace(100, 120, n)
        volume = np.full(n, 1000.0)
        result = pvt_numba(close, volume, length=1)
        valid = result[~np.isnan(result)]
        assert (np.diff(valid) > 0).all(), "PVT should increase monotonically in a rising market"


# ---------------------------------------------------------------------------
# chaikin_money_flow_numba
# ---------------------------------------------------------------------------

class TestCMF:
    def test_shape(self, ohlcv):
        result = chaikin_money_flow_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=14
        )
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 14
        result = chaikin_money_flow_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=length
        )
        assert np.isnan(result[:length - 1]).all()

    def test_values_in_range(self, ohlcv):
        """CMF ∈ [-1, 1]."""
        result = chaikin_money_flow_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=14
        )
        valid = result[~np.isnan(result)]
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_high_eq_low_does_not_crash(self):
        """When high == low, money_flow_multiplier is skipped (zero-contribution)."""
        n = 20
        price = np.full(n, 10.0, dtype=np.float64)
        volume = np.full(n, 1e5)
        result = chaikin_money_flow_numba(price, price, price, volume, length=5)
        valid = result[~np.isnan(result)]
        # All high==low → money_flow_volume stays 0 → CMF = 0
        np.testing.assert_allclose(valid, 0.0)

    def test_bullish_close_yields_positive_cmf(self):
        """Close = high → money_flow_multiplier = 1 → CMF should be positive."""
        n = 20
        low  = np.full(n, 8.0,  dtype=np.float64)
        high = np.full(n, 12.0, dtype=np.float64)
        close = high.copy()  # close at top → multiplier = 1
        volume = np.full(n, 1e5)
        result = chaikin_money_flow_numba(high, low, close, volume, length=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0)


# ---------------------------------------------------------------------------
# ad_line_numba
# ---------------------------------------------------------------------------

class TestADLine:
    def test_shape(self, ohlcv):
        result = ad_line_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert result.shape == ohlcv["close"].shape

    def test_no_nan_in_output(self, ohlcv):
        """A/D line is cumulative from index 0 — no NaN expected."""
        result = ad_line_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert not np.isnan(result).any()

    def test_starts_at_zero(self, ohlcv):
        """The A/D line initialises at 0."""
        result = ad_line_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert result[0] == 0.0

    def test_cumulative_nature(self):
        """Verify the cumulative money flow addition."""
        high  = np.array([10.0, 12.0], dtype=np.float64)
        low   = np.array([8.0,  9.0],  dtype=np.float64)
        close = np.array([9.0,  12.0], dtype=np.float64)
        volume = np.array([100.0, 200.0], dtype=np.float64)
        result = ad_line_numba(high, low, close, volume)
        # i=0: ad_line[0] = 0 (initialised)
        # i=1: mfm = ((12-9) - (12-12))/(12-9) = 3/3 = 1 → mfv = 1*200 = 200
        #       ad_line[1] = 0 + 200 = 200
        np.testing.assert_allclose(result, [0.0, 200.0])

    def test_high_eq_low_holds_previous(self):
        """high == low → money_flow is zero → A/D unchanged."""
        high  = np.array([10.0, 10.0], dtype=np.float64)
        low   = np.array([10.0, 10.0], dtype=np.float64)
        close = np.array([10.0, 10.0], dtype=np.float64)
        volume = np.array([100.0, 200.0], dtype=np.float64)
        result = ad_line_numba(high, low, close, volume)
        np.testing.assert_allclose(result, [0.0, 0.0])


# ---------------------------------------------------------------------------
# force_index_numba
# ---------------------------------------------------------------------------

class TestForceIndex:
    def test_shape(self, ohlcv):
        result = force_index_numba(ohlcv["close"], ohlcv["volume"], length=13)
        assert result.shape == ohlcv["close"].shape

    def test_no_nan_in_output(self, ohlcv):
        """EMA-smoothed force index; EMA handles leading NaNs internally."""
        result = force_index_numba(ohlcv["close"], ohlcv["volume"], length=13)
        assert not np.isnan(result).any()

    def test_rising_market_positive_force(self):
        """Rising prices with constant volume → positive force index."""
        n = 30
        close = np.linspace(100, 130, n)
        volume = np.full(n, 1000.0)
        result = force_index_numba(close, volume, length=3)
        # After first few bars, EMA of positive force → positive
        assert result[-1] > 0.0

    def test_falling_market_negative_force(self):
        """Falling prices → negative force index."""
        n = 30
        close = np.linspace(130, 100, n)
        volume = np.full(n, 1000.0)
        result = force_index_numba(close, volume, length=3)
        assert result[-1] < 0.0


# ---------------------------------------------------------------------------
# eom_numba
# ---------------------------------------------------------------------------

class TestEOM:
    def test_shape(self, ohlcv):
        result = eom_numba(ohlcv["high"], ohlcv["low"], ohlcv["volume"], length=14)
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 14
        result = eom_numba(ohlcv["high"], ohlcv["low"], ohlcv["volume"], length=length)
        assert np.isnan(result[:length - 1]).all()

    def test_zero_hl_range_returns_zero(self):
        """high == low → hl_range == 0 → eom == 0 for that bar."""
        n = 20
        price = np.full(n, 10.0, dtype=np.float64)
        volume = np.full(n, 1e5)
        result = eom_numba(price, price, volume, length=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0)

    def test_output_is_finite(self, ohlcv):
        result = eom_numba(ohlcv["high"], ohlcv["low"], ohlcv["volume"], length=14)
        valid = result[~np.isnan(result)]
        assert np.isfinite(valid).all()


# ---------------------------------------------------------------------------
# volume_profile_numba
# ---------------------------------------------------------------------------

class TestVolumeProfile:
    def test_shape(self, ohlcv):
        num_bins = 10
        result = volume_profile_numba(ohlcv["close"], ohlcv["volume"], length=20, num_bins=num_bins)
        assert result.shape == (ohlcv["n"], num_bins)

    def test_warmup_rows_are_zero(self, ohlcv):
        """First `length` rows should be all zeros (initialised, not computed)."""
        length = 20
        result = volume_profile_numba(ohlcv["close"], ohlcv["volume"], length=length, num_bins=5)
        assert (result[:length] == 0).all()

    def test_volume_not_lost_beyond_tolerance(self, ohlcv):
        """All window volume should be captured by bins."""
        length = 20
        num_bins = 10
        result = volume_profile_numba(ohlcv["close"], ohlcv["volume"], length=length, num_bins=num_bins)
        for i in range(length, ohlcv["n"]):
            expected_total = np.sum(ohlcv["volume"][i - length: i])
            actual_total = np.sum(result[i])
            np.testing.assert_allclose(
                actual_total,
                expected_total,
                rtol=1e-6,
                err_msg=f"Volume not conserved at index {i}: {actual_total} vs {expected_total}",
            )

    def test_volume_conservation_includes_max_bar(self):
        """Final bin includes the window max, so total bin volume equals total window volume."""
        n = 30
        close = 50.0 + np.arange(n, dtype=np.float64) * 0.001  # strictly increasing, no ties
        volume = np.ones(n) * 1000.0
        length = 10
        num_bins = 5
        result = volume_profile_numba(close, volume, length=length, num_bins=num_bins)
        for i in range(length, n):
            window_vol = volume[i - length: i]
            total_vol = np.sum(window_vol)
            actual_total = np.sum(result[i])
            expected = total_vol
            np.testing.assert_allclose(actual_total, expected, rtol=1e-6,
                                       err_msg=f"Unexpected bin total at index {i}")

    def test_zero_range_prices_go_to_last_bin(self):
        """Constant price is captured in the final bin due to inclusive last-edge check."""
        n = 20
        close = np.full(n, 50.0, dtype=np.float64)
        volume = np.full(n, 1000.0)
        length = 5
        result = volume_profile_numba(close, volume, length=length, num_bins=4)
        for i in range(length, n):
            np.testing.assert_allclose(np.sum(result[i]), np.sum(volume[i - length:i]), rtol=1e-6)
            assert (result[i, :3] == 0.0).all()
            np.testing.assert_allclose(result[i, 3], np.sum(volume[i - length:i]), rtol=1e-6)

    def test_non_negative_bins(self, ohlcv):
        result = volume_profile_numba(ohlcv["close"], ohlcv["volume"], length=20, num_bins=10)
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# rolling_vwap_numba
# ---------------------------------------------------------------------------

class TestRollingVWAP:
    def test_shape(self, ohlcv):
        result = rolling_vwap_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=14
        )
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 14
        result = rolling_vwap_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=length
        )
        assert np.isnan(result[:length - 1]).all()

    def test_known_value(self):
        """VWAP = Σ(TP × Volume) / Σ(Volume)."""
        high   = np.array([12.0, 11.0, 13.0], dtype=np.float64)
        low    = np.array([8.0,  9.0,  7.0],  dtype=np.float64)
        close  = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        volume = np.array([100.0, 200.0, 300.0], dtype=np.float64)

        result = rolling_vwap_numba(high, low, close, volume, length=2)

        # Window [1,2]: TP = [(11+9+10)/3, (13+7+10)/3] = [10, 10] → VWAP = 10
        np.testing.assert_allclose(result[1], 10.0)
        np.testing.assert_allclose(result[2], 10.0)

    def test_zero_volume_is_nan(self):
        """Zero volume → NaN VWAP (volume_cumsum == 0)."""
        n = 10
        price = np.full(n, 10.0, dtype=np.float64)
        volume = np.zeros(n)
        result = rolling_vwap_numba(price, price, price, volume, length=3)
        assert np.isnan(result[2:]).all()

    def test_constant_price_equals_price(self):
        """Constant price → VWAP = that price regardless of volume variation."""
        n = 20
        price = np.full(n, 50.0, dtype=np.float64)
        volume = np.random.uniform(1, 100, n)
        result = rolling_vwap_numba(price, price, price, volume, length=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 50.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# twap_numba
# ---------------------------------------------------------------------------

class TestTWAP:
    def test_shape(self, ohlcv):
        result = twap_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=14)
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 14
        result = twap_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=length)
        assert np.isnan(result[:length - 1]).all()

    def test_known_value(self):
        """TWAP = average of typical prices (H+L+C)/3 over window."""
        high  = np.array([12.0, 11.0, 13.0], dtype=np.float64)
        low   = np.array([8.0,  9.0,  7.0],  dtype=np.float64)
        close = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        result = twap_numba(high, low, close, length=2)
        # i=1: TP = [10, 10] → TWAP = 10
        np.testing.assert_allclose(result[1], 10.0)
        # i=2: TP = [10, 10] → TWAP = 10
        np.testing.assert_allclose(result[2], 10.0)

    def test_equals_rolling_typical_price_mean(self, ohlcv):
        """TWAP should match a rolling mean of typical prices from numpy."""
        length = 10
        result = twap_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=length)
        tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
        for i in range(length - 1, ohlcv["n"]):
            expected = np.mean(tp[i - length + 1:i + 1])
            np.testing.assert_allclose(result[i], expected, rtol=1e-10,
                                       err_msg=f"TWAP mismatch at index {i}")


# ---------------------------------------------------------------------------
# cci_numba
# ---------------------------------------------------------------------------

class TestCCI:
    def test_shape(self, ohlcv):
        result = cci_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=14)
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 14
        result = cci_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=length)
        assert np.isnan(result[:length - 1]).all()

    def test_flat_data_returns_zero(self):
        """Flat price → MAD = 0 → CCI = 0 (per implementation guard)."""
        n = 20
        price = np.full(n, 10.0, dtype=np.float64)
        result = cci_numba(price, price, price, length=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0)

    def test_output_is_finite(self, ohlcv):
        result = cci_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=14)
        valid = result[~np.isnan(result)]
        assert np.isfinite(valid).all()

    def test_insufficient_data_returns_all_nan(self):
        """If n < length, result is all NaN."""
        high  = np.array([10.0, 11.0], dtype=np.float64)
        low   = np.array([9.0,  10.0], dtype=np.float64)
        close = np.array([10.0, 10.5], dtype=np.float64)
        result = cci_numba(high, low, close, length=5)
        assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# average_quote_volume_numba
# ---------------------------------------------------------------------------

class TestAverageQuoteVolume:
    def test_shape(self, ohlcv):
        result = average_quote_volume_numba(ohlcv["close"], ohlcv["volume"], window_size=10)
        assert result.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        window_size = 10
        result = average_quote_volume_numba(ohlcv["close"], ohlcv["volume"], window_size=window_size)
        assert np.isnan(result[:window_size - 1]).all()

    def test_known_value(self):
        """AvgQV = mean(close[window]) × mean(volume[window])."""
        close  = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        volume = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        result = average_quote_volume_numba(close, volume, window_size=2)
        # i=1: avg_close = (10+20)/2 = 15, avg_vol = (100+200)/2 = 150 → 15*150 = 2250
        np.testing.assert_allclose(result[1], 2250.0)
        # i=2: avg_close = (20+30)/2 = 25, avg_vol = (200+300)/2 = 250 → 25*250 = 6250
        np.testing.assert_allclose(result[2], 6250.0)

    def test_constant_price_and_volume(self):
        n = 20
        close = np.full(n, 50.0)
        volume = np.full(n, 1000.0)
        result = average_quote_volume_numba(close, volume, window_size=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 50_000.0)
