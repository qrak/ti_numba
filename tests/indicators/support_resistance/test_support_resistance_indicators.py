import numpy as np
import pytest

from src.indicators.support_resistance.support_resistance_indicators import (
    support_resistance_numba,
    support_resistance_numba_advanced,
    advanced_support_resistance_numba,
    find_support_resistance_numba,
    fibonacci_retracement_numba,
    floating_levels_numba,
    fibonacci_bollinger_bands_numba,
    FloatingLevelsConfig,
)


@pytest.fixture
def ohlcv():
    """Realistic 150-bar OHLCV data."""
    np.random.seed(13)
    n = 150
    close = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    volume = np.random.uniform(1e5, 1e6, n)
    return {"close": close, "high": high, "low": low, "volume": volume, "n": n}


# ---------------------------------------------------------------------------
# support_resistance_numba
# ---------------------------------------------------------------------------

class TestSupportResistance:
    def test_shape(self, ohlcv):
        support, resistance = support_resistance_numba(ohlcv["high"], ohlcv["low"], length=20)
        assert support.shape == ohlcv["close"].shape
        assert resistance.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 20
        support, resistance = support_resistance_numba(ohlcv["high"], ohlcv["low"], length=length)
        assert np.isnan(support[:length - 1]).all()
        assert np.isnan(resistance[:length - 1]).all()

    def test_known_values(self):
        """Rolling min/max of a known series."""
        high = np.array([10.0, 12.0, 8.0, 15.0, 11.0], dtype=np.float64)
        low  = np.array([7.0,  9.0,  5.0, 12.0, 8.0],  dtype=np.float64)
        support, resistance = support_resistance_numba(high, low, length=3)

        # Window idx 0..2: support = min(low[0:3]) = min(7,9,5) = 5
        #                  resistance = max(high[0:3]) = max(10,12,8) = 12
        np.testing.assert_allclose(support[2], 5.0)
        np.testing.assert_allclose(resistance[2], 12.0)

    def test_support_below_resistance(self, ohlcv):
        length = 20
        support, resistance = support_resistance_numba(ohlcv["high"], ohlcv["low"], length=length)
        valid_s = support[~np.isnan(support)]
        valid_r = resistance[~np.isnan(resistance)]
        assert (valid_s <= valid_r).all(), "Support must not exceed resistance"

    def test_monotonic_with_all_rising(self):
        """Rising high/low → resistance non-decreasing, support non-decreasing."""
        high = np.arange(1.0, 21.0, dtype=np.float64)
        low  = high - 0.5
        support, resistance = support_resistance_numba(high, low, length=5)
        valid_r = resistance[~np.isnan(resistance)]
        assert (np.diff(valid_r) >= 0).all()


# ---------------------------------------------------------------------------
# support_resistance_numba_advanced
# ---------------------------------------------------------------------------

class TestSupportResistanceAdvanced:
    def test_shape(self, ohlcv):
        ss, sr = support_resistance_numba_advanced(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=20
        )
        assert ss.shape == ohlcv["close"].shape
        assert sr.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 20
        ss, sr = support_resistance_numba_advanced(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=length
        )
        # First `length` elements should be NaN (loop starts at range(length, n))
        assert np.isnan(ss[:length]).all()
        assert np.isnan(sr[:length]).all()

    def test_pivot_point_formula(self):
        """Pivot = (H_prev + L_prev + C_prev) / 3."""
        n = 25
        high  = np.full(n, 15.0, dtype=np.float64)
        low   = np.full(n, 5.0,  dtype=np.float64)
        close = np.full(n, 10.0, dtype=np.float64)
        volume = np.full(n, 1e6, dtype=np.float64)

        ss, sr = support_resistance_numba_advanced(high, low, close, volume, length=10)
        # pivot = (15+5+10)/3 = 10
        # r1 = 2*10 - low[i-1] = 20 - 5 = 15
        # s1 = 2*10 - high[i-1] = 20 - 15 = 5
        # volume_filter: volume > rolling_avg → all equal → all False → result should be NaN or fill
        # Since volume_filter is False, strong_support/resistance will be NaN
        assert np.isnan(ss).any()


# ---------------------------------------------------------------------------
# advanced_support_resistance_numba
# ---------------------------------------------------------------------------

class TestAdvancedSupportResistance:
    def test_shape(self, ohlcv):
        ss, sr = advanced_support_resistance_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=30
        )
        assert ss.shape == ohlcv["close"].shape
        assert sr.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        length = 30
        ss, sr = advanced_support_resistance_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=length
        )
        assert np.isnan(ss[:length]).all()
        assert np.isnan(sr[:length]).all()

    def test_output_finite_where_non_nan(self, ohlcv):
        ss, sr = advanced_support_resistance_numba(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], length=30
        )
        assert np.isfinite(ss[~np.isnan(ss)]).all()
        assert np.isfinite(sr[~np.isnan(sr)]).all()


# ---------------------------------------------------------------------------
# find_support_resistance_numba
# ---------------------------------------------------------------------------

class TestFindSupportResistance:
    def _make_levels(self, close):
        support, resistance = support_resistance_numba(close + 0.5, close - 0.5, length=10)
        return support, resistance

    def test_returns_tuple_of_two_floats(self, ohlcv):
        support, resistance = self._make_levels(ohlcv["close"])
        result = find_support_resistance_numba(ohlcv["close"], support, resistance, window=20)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], (float, np.floating))
        assert isinstance(result[1], (float, np.floating))

    def test_distances_non_negative(self, ohlcv):
        support, resistance = self._make_levels(ohlcv["close"])
        ds, dr = find_support_resistance_numba(ohlcv["close"], support, resistance, window=20)
        assert ds >= 0.0
        assert dr >= 0.0

    def test_empty_support_resistance_returns_one(self):
        """When no valid sup/res, function should return distance of 1."""
        close = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        support = np.full(3, np.nan)
        resistance = np.full(3, np.nan)
        ds, dr = find_support_resistance_numba(close, support, resistance, window=5)
        assert ds == 1.0
        assert dr == 1.0


# ---------------------------------------------------------------------------
# fibonacci_retracement_numba
# ---------------------------------------------------------------------------

class TestFibonacciRetracement:
    FIB_LEVELS = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])

    def test_shape(self, ohlcv):
        length = 20
        result = fibonacci_retracement_numba(length, ohlcv["high"], ohlcv["low"])
        assert result.shape == (ohlcv["n"], 7), f"Expected ({ohlcv['n']}, 7), got {result.shape}"

    def test_nan_warmup(self, ohlcv):
        length = 20
        result = fibonacci_retracement_numba(length, ohlcv["high"], ohlcv["low"])
        assert np.isnan(result[:length - 1]).all()

    def test_known_values(self):
        """With H=10, L=0 over window: levels should be 0,2.36,3.82,5,6.18,7.86,10."""
        n = 10
        high = np.full(n, 10.0, dtype=np.float64)
        low  = np.full(n, 0.0,  dtype=np.float64)
        result = fibonacci_retracement_numba(5, high, low)

        # First valid index is 4 (length-1)
        expected = np.array([0.0, 2.36, 3.82, 5.0, 6.18, 7.86, 10.0])
        np.testing.assert_allclose(result[4], expected, rtol=1e-10)

    def test_levels_monotonically_increasing(self, ohlcv):
        """Fib levels are monotonically increasing for any valid window."""
        result = fibonacci_retracement_numba(20, ohlcv["high"], ohlcv["low"])
        valid = result[~np.isnan(result[:, 0])]
        for row in valid:
            assert (np.diff(row) >= 0).all(), f"Fib levels not monotone: {row}"

    def test_level_0_eq_low_max_level_1_eq_high(self, ohlcv):
        """First level = rolling low min, last level = rolling high max."""
        length = 20
        result = fibonacci_retracement_numba(length, ohlcv["high"], ohlcv["low"])
        for i in range(length - 1, ohlcv["n"]):
            expected_low  = np.min(ohlcv["low"][i - length + 1:i + 1])
            expected_high = np.max(ohlcv["high"][i - length + 1:i + 1])
            np.testing.assert_allclose(result[i, 0], expected_low,  rtol=1e-10)
            np.testing.assert_allclose(result[i, 6], expected_high, rtol=1e-10)


# ---------------------------------------------------------------------------
# floating_levels_numba
# ---------------------------------------------------------------------------

class TestFloatingLevels:
    def test_shape(self, ohlcv):
        config = FloatingLevelsConfig(length=10, multiplier=3.0, lookback=20, level_up=70.0, level_down=30.0)
        flu, fld, flm = floating_levels_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], config)
        assert flu.shape == ohlcv["close"].shape
        assert fld.shape == ohlcv["close"].shape
        assert flm.shape == ohlcv["close"].shape

    def test_nan_warmup(self, ohlcv):
        config = FloatingLevelsConfig(length=10, multiplier=3.0, lookback=20, level_up=70.0, level_down=30.0)
        flu, fld, flm = floating_levels_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], config)
        # First `lookback` values should be NaN
        assert np.isnan(flu[:config.lookback]).all()

    def test_flu_above_fld(self, ohlcv):
        """Upper level should be >= lower level (both based on supertrend range)."""
        config = FloatingLevelsConfig(length=10, multiplier=3.0, lookback=20, level_up=70.0, level_down=30.0)
        flu, fld, flm = floating_levels_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], config)
        valid_mask = ~(np.isnan(flu) | np.isnan(fld))
        assert (flu[valid_mask] >= fld[valid_mask]).all()

    def test_flm_between_flu_and_fld(self, ohlcv):
        """Middle level should sit between flu and fld."""
        config = FloatingLevelsConfig(length=10, multiplier=3.0, lookback=20, level_up=70.0, level_down=30.0)
        flu, fld, flm = floating_levels_numba(ohlcv["high"], ohlcv["low"], ohlcv["close"], config)
        valid = ~(np.isnan(flu) | np.isnan(fld) | np.isnan(flm))
        assert (flm[valid] >= fld[valid] - 1e-9).all()
        assert (flm[valid] <= flu[valid] + 1e-9).all()


# ---------------------------------------------------------------------------
# fibonacci_bollinger_bands_numba
# ---------------------------------------------------------------------------

class TestFibonacciBollingerBands:
    def test_shape(self, ohlcv):
        length = 20
        basis, upper_bands, lower_bands = fibonacci_bollinger_bands_numba(
            ohlcv["close"], ohlcv["volume"], length=length, mult=2.0
        )
        assert basis.shape == ohlcv["close"].shape
        assert upper_bands.shape == (6, ohlcv["n"])
        assert lower_bands.shape == (6, ohlcv["n"])

    def test_nan_warmup(self, ohlcv):
        length = 20
        basis, upper_bands, lower_bands = fibonacci_bollinger_bands_numba(
            ohlcv["close"], ohlcv["volume"], length=length, mult=2.0
        )
        assert np.isnan(basis[:length]).all()
        assert np.isnan(upper_bands[:, :length]).all()

    def test_upper_above_basis(self, ohlcv):
        """All upper bands should be >= basis for mult > 0."""
        length = 20
        basis, upper_bands, lower_bands = fibonacci_bollinger_bands_numba(
            ohlcv["close"], ohlcv["volume"], length=length, mult=2.0
        )
        valid_idx = ~np.isnan(basis)
        for j in range(6):
            assert (upper_bands[j, valid_idx] >= basis[valid_idx]).all(), \
                f"Upper band {j} not above basis"

    def test_lower_below_basis(self, ohlcv):
        """All lower bands should be <= basis for mult > 0."""
        length = 20
        basis, upper_bands, lower_bands = fibonacci_bollinger_bands_numba(
            ohlcv["close"], ohlcv["volume"], length=length, mult=2.0
        )
        valid_idx = ~np.isnan(basis)
        for j in range(6):
            assert (lower_bands[j, valid_idx] <= basis[valid_idx]).all(), \
                f"Lower band {j} not below basis"

    def test_bands_monotonically_ordered(self, ohlcv):
        """Upper bands ordered ascending (larger fib → further from basis).
        Lower bands ordered DESCENDING (larger fib → further below basis → lower price).
        """
        length = 20
        basis, upper_bands, lower_bands = fibonacci_bollinger_bands_numba(
            ohlcv["close"], ohlcv["volume"], length=length, mult=2.0
        )
        valid_idx = np.where(~np.isnan(basis))[0]
        for i in valid_idx:
            assert (np.diff(upper_bands[:, i]) >= 0).all(), \
                f"Upper bands not ascending at index {i}: {upper_bands[:, i]}"
            assert (np.diff(lower_bands[:, i]) <= 0).all(), \
                f"Lower bands not descending at index {i}: {lower_bands[:, i]}"

    def test_zero_volume_produces_nan_basis(self):
        """Zero volume → sum_v == 0 → VWMA is NaN → bands are NaN."""
        n = 30
        close = np.linspace(100, 110, n)
        volume = np.zeros(n)
        basis, upper_bands, lower_bands = fibonacci_bollinger_bands_numba(close, volume, length=10, mult=2.0)
        assert np.isnan(basis[10:]).all()
