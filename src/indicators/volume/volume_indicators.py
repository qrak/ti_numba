import numpy as np
from src.indicators.overlap import ema_numba
from numba import njit


@njit(cache=True)
def cci_numba(high, low, close, length=14, c=0.015):
    n = len(close)
    cci = np.full(n, np.nan)

    if n < length:
        return cci

    tp = (high + low + close) / 3.0

    # FIXED: Converted slow slice mean to O(N) running sum for the mean
    sum_tp = np.sum(tp[:length])
    mean_tp = sum_tp / length

    mad_sum = 0.0
    for j in range(length):
        mad_sum += abs(tp[j] - mean_tp)
    mad_tp = mad_sum / length

    if mad_tp == 0:
        cci[length - 1] = 0.0
    else:
        cci[length - 1] = (tp[length - 1] - mean_tp) / (c * mad_tp)

    for i in range(length, n):
        old_val = tp[i - length]
        new_val = tp[i]

        sum_tp += new_val - old_val
        mean_tp = sum_tp / length

        mad_sum = 0.0
        for j in range(i - length + 1, i + 1):
            mad_sum += abs(tp[j] - mean_tp)
        mad_tp = mad_sum / length

        if mad_tp == 0:
            cci[i] = 0.0
        else:
            cci[i] = (tp[i] - mean_tp) / (c * mad_tp)

    return cci

@njit(cache=True)
def mfi_numba(high, low, close, volume, length=14, drift=1):
    n = len(high)
    mfi = np.full(n, np.nan)

    tp = np.zeros(n)
    rmf = np.zeros(n)

    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3
        rmf[i] = tp[i] * volume[i]

    for i in range(length, n):
        pmf = 0
        nmf = 0

        for j in range(i - length + 1, i + 1):
            tp_diff = tp[j] - ((high[j - drift] + low[j - drift] + close[j - drift]) / 3)

            if tp_diff > 0:
                pmf += rmf[j]
            elif tp_diff < 0:
                nmf += rmf[j]

        if nmf == 0:
            mfi[i] = 100
        else:
            mfr = pmf / nmf
            mfi[i] = 100 * mfr / (1 + mfr)

    return mfi

@njit(cache=True)
def obv_numba(close, volume, length, initial=1):
    n = len(close)
    obv = np.full(n, np.nan)

    obv[length - 1] = initial * volume[length - 1]

    for i in range(length, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    return obv

@njit(cache=True)
def pvt_numba(close, volume, length, drift=1):
    n = len(close)
    pvt = np.full(n, np.nan)
    pv = 0

    for i in range(length - 1, n):
        roc = (close[i] - close[i - drift]) * (1 / close[i - drift])
        pv += roc * volume[i]
        pvt[i] = pv

    return pvt

@njit(cache=True)
def chaikin_money_flow_numba(high, low, close, volume, length):
    n = len(close)
    cmf = np.full(n, np.nan)

    for i in range(length - 1, n):
        money_flow_volume = 0
        volume_sum = 0

        for j in range(i - length + 1, i + 1):
            if high[j] != low[j]:
                money_flow_multiplier = ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
                money_flow_volume += money_flow_multiplier * volume[j]
            volume_sum += volume[j]

        if volume_sum != 0:
            cmf[i] = money_flow_volume / volume_sum

    return cmf

@njit(cache=True)
def ad_line_numba(high, low, close, volume):
    n = len(close)
    ad_line = np.zeros(n)

    for i in range(1, n):
        if high[i] != low[i]:
            money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            money_flow_volume = money_flow_multiplier * volume[i]
            ad_line[i] = ad_line[i - 1] + money_flow_volume
        else:
            ad_line[i] = ad_line[i - 1]

    return ad_line

@njit(cache=True)
def force_index_numba(close, volume, length):
    n = len(close)
    force_index = np.zeros(n)

    force_index[0] = 0

    for i in range(1, n):
        force_index[i] = (close[i] - close[i - 1]) * volume[i]

    force_index_ema = ema_numba(force_index, length)

    return force_index_ema

@njit(cache=True)
def eom_numba(high, low, volume, length=14, divisor=10000.0, drift=1):
    n = len(high)
    eom = np.full(n, np.nan)
    eom_sma = np.full(n, np.nan)

    for i in range(drift, n):
        hl_range = high[i] - low[i]
        if hl_range == 0:
            eom[i] = 0
        else:
            distance = ((high[i] + low[i]) / 2) - ((high[i - drift] + low[i - drift]) / 2)
            box_ratio = (volume[i] / divisor) / hl_range
            eom[i] = distance / box_ratio if box_ratio != 0 else 0

    if n >= length + drift:
        window_sum = 0.0
        for i in range(drift, length + drift):
            window_sum += eom[i]

        eom_sma[length + drift - 1] = window_sum / length

        for i in range(length + drift, n):
            window_sum += eom[i] - eom[i - length]
            eom_sma[i] = window_sum / length

    return eom_sma

@njit(cache=True)
def volume_profile_numba(close, volume, length, num_bins):
    n = len(close)
    result = np.zeros((n, num_bins))

    for i in range(length, n):
        window_close = close[i - length:i]
        window_volume = volume[i - length:i]

        price_range = np.linspace(np.min(window_close), np.max(window_close), num_bins + 1)
        volume_profile = np.zeros(num_bins)

        for j in range(num_bins):
            mask = (window_close >= price_range[j]) & (window_close < price_range[j + 1])
            volume_profile[j] = np.sum(window_volume[mask])

        result[i] = volume_profile

    return result

@njit(cache=True)
def rolling_vwap_numba(high, low, close, volume, length):
    n = len(high)
    vwap = np.full(n, np.nan)
    tpv_cumsum = 0
    volume_cumsum = 0

    for i in range(n):
        tp = (high[i] + low[i] + close[i]) / 3
        tpv = tp * volume[i]
        tpv_cumsum += tpv
        volume_cumsum += volume[i]

        if i >= length:
            old_tp = (high[i-length] + low[i-length] + close[i-length]) / 3
            old_tpv = old_tp * volume[i-length]
            tpv_cumsum -= old_tpv
            volume_cumsum -= volume[i-length]

        if i >= length - 1 and volume_cumsum != 0:
            vwap[i] = tpv_cumsum / volume_cumsum

    return vwap

@njit(cache=True)
def twap_numba(high, low, close, length):
    n = len(high)
    twap = np.full(n, np.nan)

    if n < length:
        return twap

    tp_sum = 0.0
    for i in range(length):
        tp = (high[i] + low[i] + close[i]) / 3
        tp_sum += tp

    twap[length - 1] = tp_sum / length

    for i in range(length, n):
        new_tp = (high[i] + low[i] + close[i]) / 3
        old_tp = (high[i - length] + low[i - length] + close[i - length]) / 3
        tp_sum += new_tp - old_tp
        twap[i] = tp_sum / length

    return twap

@njit(cache=True)
def average_quote_volume_numba(close_prices, volumes, window_size):
    n = len(close_prices)
    quote_volumes = np.full(n, np.nan)

    if n < window_size:
        return quote_volumes

    close_sum = np.sum(close_prices[:window_size])
    vol_sum = np.sum(volumes[:window_size])

    quote_volumes[window_size - 1] = (close_sum / window_size) * (vol_sum / window_size)

    for i in range(window_size, n):
        close_sum += close_prices[i] - close_prices[i - window_size]
        vol_sum += volumes[i] - volumes[i - window_size]
        quote_volumes[i] = (close_sum / window_size) * (vol_sum / window_size)

    return quote_volumes

