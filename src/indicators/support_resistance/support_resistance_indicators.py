import numpy as np
from src.indicators.trend import supertrend_numba
from numba import njit


@njit(cache=True)
def support_resistance_numba(high, low, length):
    n = len(high)
    rolling_resistance = np.full(n, np.nan)
    rolling_support = np.full(n, np.nan)

    for i in range(length - 1, n):
        rolling_resistance[i] = np.max(high[i - length + 1:i + 1])
        rolling_support[i] = np.min(low[i - length + 1:i + 1])

    return rolling_support, rolling_resistance


@njit(cache=True)
def support_resistance_numba_advanced(high, low, close, volume, length):
    n = len(close)
    pivot_points = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    volume_filter = np.full(n, False)
    rolling_avg_volume = np.full(n, np.nan)

    if n >= length:
        vol_sum = np.sum(volume[:length])

        for i in range(length, n):
            # FIXED: Converted slow slice mean to O(N) running sum
            rolling_avg_volume[i] = vol_sum / length

            pivot_points[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

            r1[i] = (2 * pivot_points[i]) - low[i - 1]
            s1[i] = (2 * pivot_points[i]) - high[i - 1]

            volume_filter[i] = volume[i] > rolling_avg_volume[i]

            vol_sum += volume[i] - volume[i - length]

    strong_support = np.where(volume_filter, s1, np.nan)
    strong_resistance = np.where(volume_filter, r1, np.nan)

    return strong_support, strong_resistance



@njit(cache=True)
def advanced_support_resistance_numba(high, low, close, volume, length=50, strength_threshold=2, persistence=1,
                                      volume_factor=2.0, price_factor=0.005):
    n = len(close)
    pivot_points = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    volume_filter = np.full(n, False)
    rolling_avg_volume = np.full(n, np.nan)

    support_strength = np.zeros(n)
    resistance_strength = np.zeros(n)

    strong_support = np.full(n, np.nan)
    strong_resistance = np.full(n, np.nan)

    if n >= length:
        vol_sum = np.sum(volume[:length])

        for i in range(length, n):
            # FIXED: Converted slow slice mean to O(N) running sum
            rolling_avg_volume[i] = vol_sum / length
            pivot_points[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3

            r1[i] = (2 * pivot_points[i]) - low[i - 1]
            s1[i] = (2 * pivot_points[i]) - high[i - 1]
            r2[i] = pivot_points[i] + (high[i - 1] - low[i - 1])
            s2[i] = pivot_points[i] - (high[i - 1] - low[i - 1])

            volume_filter[i] = volume[i] > rolling_avg_volume[i]

            if close[i] < s1[i]:
                support_strength[i] = support_strength[i - 1] + 1
            elif close[i] > r1[i]:
                resistance_strength[i] = resistance_strength[i - 1] + 1
            else:
                support_strength[i] = max(0, support_strength[i - 1] - 1)
                resistance_strength[i] = max(0, resistance_strength[i - 1] - 1)

            if volume_filter[i] and volume[i] > volume_factor * rolling_avg_volume[i]:
                if support_strength[i] >= strength_threshold and close[i] < (1 - price_factor) * s1[i]:
                    for j in range(max(0, i - persistence + 1), i + 1):
                        strong_support[j] = min(s1[j], s2[j])
                if resistance_strength[i] >= strength_threshold and close[i] > (1 + price_factor) * r1[i]:
                    for j in range(max(0, i - persistence + 1), i + 1):
                        strong_resistance[j] = max(r1[j], r2[j])

            vol_sum += volume[i] - volume[i - length]

    return strong_support, strong_resistance


@njit(cache=True)
def find_support_resistance_numba(close, support, resistance, window):
    current_price = close[-1]

    valid_support = support[~np.isnan(support)][-window:]
    valid_resistance = resistance[~np.isnan(resistance)][-window:]

    if len(valid_support) > 0:
        nearest_support = np.max(valid_support[valid_support < current_price]) if np.any(
            valid_support < current_price) else np.min(valid_support)
        distance_to_support = (current_price - nearest_support) / current_price
    else:
        distance_to_support = 1

    if len(valid_resistance) > 0:
        nearest_resistance = np.min(valid_resistance[valid_resistance > current_price]) if np.any(
            valid_resistance > current_price) else np.max(valid_resistance)
        distance_to_resistance = (nearest_resistance - current_price) / current_price
    else:
        distance_to_resistance = 1

    return distance_to_support, distance_to_resistance

@njit(cache=True)
def fibonacci_retracement_numba(length, high, low):
    n = len(high)
    fib_levels = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])

    retracement_values = np.full((n, len(fib_levels)), np.nan)

    for i in range(length - 1, n):
        high_max = np.max(high[i - length + 1:i + 1])
        low_min = np.min(low[i - length + 1:i + 1])
        diff = high_max - low_min

        for j, level in enumerate(fib_levels):
            retracement_values[i, j] = low_min + diff * level

    return retracement_values

from typing import NamedTuple

class FloatingLevelsConfig(NamedTuple):
    length: int
    multiplier: float
    lookback: int
    level_up: float
    level_down: float

@njit(cache=True)
def floating_levels_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         config: FloatingLevelsConfig):
    supertrend, _ = supertrend_numba(high, low, close, config.length, config.multiplier)
    n = len(supertrend)
    flu = np.empty(n, dtype=np.float64)
    fld = np.empty(n, dtype=np.float64)
    flm = np.empty(n, dtype=np.float64)

    for i in range(config.lookback, n):
        mini = np.min(supertrend[i - config.lookback:i])
        maxi = np.max(supertrend[i - config.lookback:i])
        rrange = maxi - mini
        flu[i] = mini + config.level_up * rrange / 100.0
        fld[i] = mini + config.level_down * rrange / 100.0
        flm[i] = mini + 0.5 * rrange

    flu[:config.lookback] = np.nan
    fld[:config.lookback] = np.nan
    flm[:config.lookback] = np.nan

    return flu, fld, flm

@njit(cache=True)
def fibonacci_bollinger_bands_numba(src, volume, length, mult):
    n = len(src)
    vwma_values = np.empty(n, dtype=np.float64)
    stdev_values = np.empty(n, dtype=np.float64)
    basis = np.empty(n, dtype=np.float64)
    dev = np.empty(n, dtype=np.float64)
    upper_bands = np.empty((6, n), dtype=np.float64)
    lower_bands = np.empty((6, n), dtype=np.float64)
    fib_levels = np.array([0.236, 0.382, 0.5, 0.618, 0.764, 1.0], dtype=np.float64)

    # Fill NaNs for the initial window
    vwma_values[:length - 1] = np.nan
    stdev_values[:length - 1] = np.nan
    basis[:length - 1] = np.nan
    dev[:length - 1] = np.nan
    for j in range(6):
        upper_bands[j, :length - 1] = np.nan
        lower_bands[j, :length - 1] = np.nan

    if n < length:
        return basis, upper_bands, lower_bands

    # Initialize running sums for the first complete window
    sum_pv = 0.0
    sum_v = 0.0
    sum_src = 0.0
    sum_src_sq = 0.0

    for i in range(length):
        pv = src[i] * volume[i]
        sum_pv += pv
        sum_v += volume[i]
        sum_src += src[i]
        sum_src_sq += src[i] ** 2

    # Calculate for the first valid index (length - 1)
    vwma_values[length - 1] = sum_pv / sum_v if sum_v != 0 else np.nan
    mean = sum_src / length
    variance = (sum_src_sq / length) - (mean * mean)
    stdev_values[length - 1] = np.sqrt(max(0.0, variance))

    basis[length - 1] = vwma_values[length - 1]
    dev[length - 1] = mult * stdev_values[length - 1]

    for j in range(6):
        upper_bands[j, length - 1] = basis[length - 1] + (fib_levels[j] * dev[length - 1])
        lower_bands[j, length - 1] = basis[length - 1] - (fib_levels[j] * dev[length - 1])

    # O(N) rolling calculation for the rest
    for i in range(length, n):
        # Subtract outgoing value
        old_idx = i - length
        sum_pv -= src[old_idx] * volume[old_idx]
        sum_v -= volume[old_idx]
        sum_src -= src[old_idx]
        sum_src_sq -= src[old_idx] ** 2

        # Add incoming value
        sum_pv += src[i] * volume[i]
        sum_v += volume[i]
        sum_src += src[i]
        sum_src_sq += src[i] ** 2

        # Calculate current values
        vwma_values[i] = sum_pv / sum_v if sum_v != 0 else np.nan
        mean = sum_src / length
        variance = (sum_src_sq / length) - (mean * mean)
        stdev_values[i] = np.sqrt(max(0.0, variance))

        basis[i] = vwma_values[i]
        dev[i] = mult * stdev_values[i]

        for j in range(6):
            upper_bands[j, i] = basis[i] + (fib_levels[j] * dev[i])
            lower_bands[j, i] = basis[i] - (fib_levels[j] * dev[i])

    return basis, upper_bands, lower_bands
