import numpy as np
from numba import njit

@njit(cache=True)
def apa_adaptive_eot_numba(closeprices, q1_=0.8, q2_=0.4, minlen=10, maxlen=48, avelen=3):
    masterdom = _auto_dom_imp(closeprices, minlen, maxlen, avelen)
    dcout = max(minlen, min(maxlen, int(round(masterdom))))

    qup = _eot(closeprices, dcout, q1_)
    qdn = _eot(closeprices, dcout, q2_)

    return qup, qdn

@njit(cache=True)
def _eot(closeprices, lpperiod, k):
    n = len(closeprices)
    pk = np.zeros(n)
    filt = _f_ess(_f_hp(closeprices, lpperiod), lpperiod)
    x = np.zeros(n)
    q = np.zeros(n)

    for i in range(1, n):
        pk[i] = np.maximum(abs(filt[i]), 0.99 * pk[i - 1])
        x[i] = filt[i] / pk[i] if pk[i] != 0 else 0
        q[i] = (x[i] + k) / (k * x[i] + 1) if x[i] != 0 else np.nan

    return q

@njit(cache=True)
def _f_ess(source, length_):
    s = 1.414
    a = np.exp(-s * np.pi / length_)
    b = 2 * a * np.cos(s * np.pi / length_)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3
    out = np.zeros_like(source)
    for i in range(2, len(source)):
        out[i] = c1 * (source[i] + source[i - 1]) / 2 + c2 * out[i - 1] + c3 * out[i - 2]
    return out

@njit(cache=True)
def _f_hp(source, maxlen):
    c = 360 * np.pi / 180
    alpha = (1 - np.sin(c / maxlen)) / np.cos(c / maxlen)
    hp = np.zeros_like(source)
    for i in range(1, len(source)):
        hp[i] = 0.5 * (1 + alpha) * (source[i] - source[i - 1]) + alpha * hp[i - 1]
    return hp


@njit(cache=True)
def _auto_dom_imp(source, minlen, maxlen, avelen):
    c = 2 * np.pi
    filt = _f_ess(_f_hp(source, maxlen), minlen)
    arr_size = maxlen * 2
    corr = np.zeros(arr_size)
    cospart = np.zeros(arr_size)
    sinpart = np.zeros(arr_size)
    sqsum = np.zeros(arr_size)
    r1 = np.zeros(arr_size)
    r2 = np.zeros(arr_size)
    pwr = np.zeros(arr_size)

    for lag in range(maxlen):
        m = avelen if avelen != 0 else lag
        sx, sy, sxx, syy, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(m):
            x = filt[i]
            y = filt[lag + i]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
        if (m * sxx - sx * sx) * (m * syy - sy * sy) > 0:
            corr[lag] = (m * sxy - sx * sy) / np.sqrt((m * sxx - sx * sx) * (m * syy - sy * sy))

    for period in range(minlen, maxlen):
        cospart[period] = 0
        sinpart[period] = 0
        for n in range(avelen, maxlen):
            cospart[period] += corr[n] * np.cos(c * n / period)
            sinpart[period] += corr[n] * np.sin(c * n / period)
        sqsum[period] = cospart[period] ** 2 + sinpart[period] ** 2

    for period in range(minlen, maxlen):
        r2[period] = r1[period]
        r1[period] = 0.2 * sqsum[period] ** 2 + 0.8 * r2[period]

    maxpwr = np.max(r1[minlen:maxlen])

    if maxpwr == 0:
        return 1

    for period in range(avelen, maxlen):
        pwr[period] = r1[period] / maxpwr

    peakpwr = np.max(pwr[minlen:maxlen])
    spx, sp = 0.0, 0.0

    for period in range(minlen, maxlen):
        if pwr[period] >= 0.5:
            spx += period * pwr[period]
            sp += pwr[period]

    for period in range(minlen, maxlen):
        if peakpwr >= 0.25 and pwr[period] >= 0.25:
            spx += period * pwr[period]
            sp += pwr[period]

    dominantcycle = spx / sp if sp != 0 else 0
    dominantcycle = dominantcycle if sp >= 0.25 else dominantcycle
    dominantcycle = max(dominantcycle, 1)
    return dominantcycle

@njit(cache=True)
def kurtosis_numba(arr, length):
    n = len(arr)
    kurtosis_values = np.full(n, np.nan)

    if length < 4 or n < length:
        return kurtosis_values

    # FIXED: Converted slow slice sum to O(N) running sum
    sum_x = np.sum(arr[:length])
    sum_x2 = np.sum(arr[:length] ** 2)
    sum_x3 = np.sum(arr[:length] ** 3)
    sum_x4 = np.sum(arr[:length] ** 4)

    length_reciprocal = 1.0 / length
    kurtosis_constant = (length * (length + 1)) / ((length - 1) * (length - 2) * (length - 3))

    mean = sum_x * length_reciprocal
    variance = (sum_x2 - sum_x * sum_x * length_reciprocal) / (length - 1)
    std_dev = np.sqrt(max(0.0, variance))

    if std_dev > 0:
        m2 = mean * mean
        m3 = m2 * mean
        m4 = m3 * mean
        sum_diff4 = sum_x4 - 4 * mean * sum_x3 + 6 * m2 * sum_x2 - 4 * m3 * sum_x + length * m4
        kurtosis_sum = sum_diff4 / (std_dev ** 4)
        kurtosis = kurtosis_constant * kurtosis_sum - (3 * ((length - 1) ** 2) / ((length - 2) * (length - 3)))
        kurtosis_values[length - 1] = kurtosis

    for i in range(length, n):
        old_val = arr[i - length]
        new_val = arr[i]

        sum_x += new_val - old_val
        sum_x2 += new_val ** 2 - old_val ** 2
        sum_x3 += new_val ** 3 - old_val ** 3
        sum_x4 += new_val ** 4 - old_val ** 4

        mean = sum_x * length_reciprocal
        variance = (sum_x2 - sum_x * sum_x * length_reciprocal) / (length - 1)
        std_dev = np.sqrt(max(0.0, variance))

        if std_dev > 0:
            m2 = mean * mean
            m3 = m2 * mean
            m4 = m3 * mean
            sum_diff4 = sum_x4 - 4 * mean * sum_x3 + 6 * m2 * sum_x2 - 4 * m3 * sum_x + length * m4
            kurtosis_sum = sum_diff4 / (std_dev ** 4)
            kurtosis = kurtosis_constant * kurtosis_sum - (3 * ((length - 1) ** 2) / ((length - 2) * (length - 3)))
            kurtosis_values[i] = kurtosis

    return kurtosis_values

@njit(cache=True)
def skew_numba(close, length=30):
    n = len(close)
    skew_values = np.full(n, np.nan)

    if length < 3 or n < length:
        return skew_values

    # FIXED: Converted slow slice sum to O(N) running sum
    sum_x = np.sum(close[:length])
    sum_x2 = np.sum(close[:length] ** 2)
    sum_x3 = np.sum(close[:length] ** 3)

    mean = sum_x / length
    variance = (sum_x2 - (sum_x ** 2) / length) / (length - 1)
    std_dev = np.sqrt(max(0.0, variance))

    if std_dev > 0:
        m2 = mean * mean
        m3 = m2 * mean
        sum_diff3 = sum_x3 - 3 * mean * sum_x2 + 3 * m2 * sum_x - length * m3
        skew_sum = sum_diff3 / (std_dev ** 3)
        skew_values[length - 1] = (length / ((length - 1) * (length - 2))) * skew_sum

    for i in range(length, n):
        old_val = close[i - length]
        new_val = close[i]

        sum_x += new_val - old_val
        sum_x2 += new_val ** 2 - old_val ** 2
        sum_x3 += new_val ** 3 - old_val ** 3

        mean = sum_x / length
        variance = (sum_x2 - (sum_x ** 2) / length) / (length - 1)
        std_dev = np.sqrt(max(0.0, variance))

        if std_dev > 0:
            m2 = mean * mean
            m3 = m2 * mean
            sum_diff3 = sum_x3 - 3 * mean * sum_x2 + 3 * m2 * sum_x - length * m3
            skew_sum = sum_diff3 / (std_dev ** 3)
            skew_values[i] = (length / ((length - 1) * (length - 2))) * skew_sum

    return skew_values

@njit(cache=True)
def stdev_numba(close, length=30, ddof=1):
    variance_values = variance_numba(close, length, ddof)
    return np.sqrt(variance_values)

@njit(cache=True)
def variance_numba(close, length=30, ddof=1):
    n = len(close)
    variance_values = np.full(n, np.nan)

    if n < length:
        return variance_values

    # FIXED: Converted slow slice sum to O(N) running sum
    sum_x = np.sum(close[:length])
    sum_x2 = np.sum(close[:length] ** 2)

    variance = (sum_x2 - (sum_x ** 2) / length) / (length - ddof)
    variance_values[length - 1] = max(0.0, variance)

    for i in range(length, n):
        sum_x += close[i] - close[i - length]
        sum_x2 += close[i] ** 2 - close[i - length] ** 2

        variance = (sum_x2 - (sum_x ** 2) / length) / (length - ddof)
        variance_values[i] = max(0.0, variance)

    return variance_values

@njit(cache=True)
def zscore_numba(close, length=30, std=1.0):
    n = len(close)
    zscore_values = np.full(n, np.nan)

    if n < length:
        return zscore_values

    # FIXED: Converted slow slice sum to O(N) running sum
    sum_x = np.sum(close[:length])
    sum_x2 = np.sum(close[:length] ** 2)

    mean = sum_x / length
    variance = (sum_x2 - (sum_x ** 2) / length) / (length - 1)
    stdev = np.sqrt(max(0.0, variance))

    if stdev != 0:
        zscore_values[length - 1] = (close[length - 1] - mean) / (std * stdev)

    for i in range(length, n):
        sum_x += close[i] - close[i - length]
        sum_x2 += close[i] ** 2 - close[i - length] ** 2

        mean = sum_x / length
        variance = (sum_x2 - (sum_x ** 2) / length) / (length - 1)
        stdev = np.sqrt(max(0.0, variance))

        if stdev != 0:
            zscore_values[i] = (close[i] - mean) / (std * stdev)

    return zscore_values

@njit(cache=True)
def mad_numba(close, length=30):
    n = len(close)
    mad_values = np.full(n, np.nan)

    if n < length:
        return mad_values

    # FIXED: Converted slow slice mean to O(N) running sum for the mean
    sum_x = np.sum(close[:length])
    mean = sum_x / length

    mad_sum = 0.0
    for j in range(length):
        mad_sum += abs(close[j] - mean)
    mad_values[length - 1] = mad_sum / length

    for i in range(length, n):
        sum_x += close[i] - close[i - length]
        mean = sum_x / length

        mad_sum = 0.0
        for j in range(i - length + 1, i + 1):
            mad_sum += abs(close[j] - mean)
        mad_values[i] = mad_sum / length

    return mad_values

@njit(cache=True)
def quantile_numba(close, length=30, q=0.5):
    n = len(close)
    quantile_values = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        quantile_values[i] = np.quantile(window, q)

    return quantile_values

@njit(cache=True)
def entropy_numba(close, length=10, base=2.0):
    n = len(close)
    entropy = np.full(n, np.nan)

    if length > n:
        return entropy

    log_base = np.log(base)  # precompute log base

    total = 0.0
    sum_clogc = 0.0

    # FIXED: Converted O(N*K) slice sums to O(N) running sums
    for j in range(length):
        c = close[j]
        total += c
        if c > 0:
            sum_clogc += c * np.log(c)

    for i in range(length, n):
        if total > 0:
            ent = -( (sum_clogc / total) - np.log(total) ) / log_base
            entropy[i] = ent

        old_c = close[i - length]
        new_c = close[i]

        total += new_c - old_c

        if new_c > 0:
            sum_clogc += new_c * np.log(new_c)
        if old_c > 0:
            sum_clogc -= old_c * np.log(old_c)

    return entropy

@njit(cache=True)
def hurst_numba(ts: np.ndarray, max_lag: int = 20) -> np.ndarray:
    n = len(ts)
    hurst_values = np.full(n, np.nan, dtype=np.float64)
    
    if n <= max_lag + 2:
        return hurst_values

    lags = np.arange(2, max_lag)
    num_lags = len(lags)

    # FIXED: Converted O(N^2) expanding window calculation to O(N) running sum
    sum_diff_sq = np.zeros(num_lags, dtype=np.float64)
    counts = np.zeros(num_lags, dtype=np.float64)

    # Initialize running sums up to max_lag + 1
    for j in range(num_lags):
        lag = lags[j]
        for idx in range(lag, max_lag + 2):
            diff = ts[idx] - ts[idx - lag]
            sum_diff_sq[j] += diff * diff
            counts[j] += 1

    for i in range(max_lag + 2, n):
        log_lags_sum = 0.0
        log_tau_sum = 0.0
        log_lags_sq_sum = 0.0
        log_lags_tau_sum = 0.0
        n_points = 0
        
        for j in range(num_lags):
            lag = lags[j]
            diff = ts[i] - ts[i - lag]
            sum_diff_sq[j] += diff * diff
            counts[j] += 1
            
            tau = 0.0
            if counts[j] > 0:
                tau = np.sqrt(sum_diff_sq[j] / counts[j])

            if tau > 0:
                log_l = np.log(lag)
                log_t = np.log(tau)
                log_lags_sum += log_l
                log_tau_sum += log_t
                log_lags_sq_sum += log_l * log_l
                log_lags_tau_sum += log_l * log_t
                n_points += 1

        if n_points >= 2:
            denominator = n_points * log_lags_sq_sum - log_lags_sum * log_lags_sum
            if denominator != 0:
                slope = (n_points * log_lags_tau_sum - log_lags_sum * log_tau_sum) / denominator
                hurst_values[i] = slope

    return hurst_values

@njit(cache=True)
def linreg_numba(close, length=14, r=False):
    n = len(close)
    linreg = np.full(n, np.nan)

    if n < length:
        return linreg

    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    # FIXED: Converted O(N*K) slice sums to O(N) running sums
    y_sum = 0.0
    xy_sum = 0.0
    y2_sum = 0.0

    for j in range(length):
        val = close[j]
        y_sum += val
        xy_sum += (j + 1) * val
        if r:
            y2_sum += val * val

    m = (length * xy_sum - x_sum * y_sum) / divisor

    if r:
        rn = length * xy_sum - x_sum * y_sum
        rd_val = divisor * (length * y2_sum - y_sum * y_sum)
        rd = np.sqrt(max(0.0, rd_val))
        linreg[length - 1] = rn / rd if rd != 0 else 0.0
    else:
        linreg[length - 1] = m

    for i in range(length, n):
        old_val = close[i - length]
        new_val = close[i]

        xy_sum = xy_sum - y_sum + length * new_val
        y_sum += new_val - old_val

        if r:
            y2_sum += new_val * new_val - old_val * old_val

        m = (length * xy_sum - x_sum * y_sum) / divisor

        if r:
            rn = length * xy_sum - x_sum * y_sum
            rd_val = divisor * (length * y2_sum - y_sum * y_sum)
            rd = np.sqrt(max(0.0, rd_val))
            linreg[i] = rn / rd if rd != 0 else 0.0
        else:
            linreg[i] = m

    return linreg

@njit(cache=True)
def calculate_eot_numba(close_prices, period=21, q1=0.8, q2=0.4):
    angle = 0.707 * 2 * np.pi / 100
    alpha1 = (np.cos(angle) + np.sin(angle) - 1) / np.cos(angle)
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    n = len(close_prices)
    hp = np.zeros(n)
    filt = np.zeros(n)
    pk = np.zeros(n)
    x = np.zeros(n)
    quotient1 = np.full(n, np.nan)
    quotient2 = np.full(n, np.nan)

    for i in range(period, n):
        hp[i] = (1 - alpha1 / 2) ** 2 * (close_prices[i] - 2 * close_prices[i - 1] + close_prices[i - 2]) + \
                2 * (1 - alpha1) * hp[i - 1] - (1 - alpha1) ** 2 * hp[i - 2]
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
        if abs(filt[i]) > 0.991 * pk[i - 1]:
            pk[i] = abs(filt[i])
        else:
            pk[i] = 0.991 * pk[i - 1]
        x[i] = filt[i] / pk[i] if pk[i] != 0 else 0
        quotient1[i] = (x[i] + q1) / (q1 * x[i] + 1) if x[i] != 0 else np.nan
        quotient2[i] = (x[i] + q2) / (q2 * x[i] + 1) if x[i] != 0 else np.nan

    return quotient1, quotient2
