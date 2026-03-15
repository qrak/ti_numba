import numpy as np
from src.indicators.volume.volume_indicators import mfi_numba, chaikin_money_flow_numba

# Original functions directly copied or imported
mfi_orig = mfi_numba
cmf_orig = chaikin_money_flow_numba

from numba import njit

@njit(cache=True)
def mfi_opt(high, low, close, volume, length=14, drift=1):
    n = len(high)
    mfi = np.full(n, np.nan)

    if n < length:
        return mfi

    tp = np.zeros(n)
    rmf = np.zeros(n)
    pmf_arr = np.zeros(n)
    nmf_arr = np.zeros(n)

    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3
        rmf[i] = tp[i] * volume[i]

    for i in range(drift, n):
        tp_diff = tp[i] - tp[i - drift]
        if tp_diff > 0:
            pmf_arr[i] = rmf[i]
        elif tp_diff < 0:
            nmf_arr[i] = rmf[i]

    # The original loop:
    # for i in range(length, n):
    #     for j in range(i - length + 1, i + 1):
    #         ...
    # So if i = length, j goes from 1 to length

    if length < n:
        pmf = 0.0
        nmf = 0.0
        for j in range(1, length + 1):
            pmf += pmf_arr[j]
            nmf += nmf_arr[j]

        if nmf == 0:
            mfi[length] = 100.0
        else:
            mfr = pmf / nmf
            mfi[length] = 100.0 * mfr / (1.0 + mfr)

        for i in range(length + 1, n):
            pmf += pmf_arr[i] - pmf_arr[i - length]
            nmf += nmf_arr[i] - nmf_arr[i - length]

            if nmf == 0:
                mfi[i] = 100.0
            else:
                mfr = pmf / nmf
                mfi[i] = 100.0 * mfr / (1.0 + mfr)

    return mfi

@njit(cache=True)
def cmf_opt(high, low, close, volume, length):
    n = len(close)
    cmf = np.full(n, np.nan)

    if n < length:
        return cmf

    mfv_arr = np.zeros(n)
    for i in range(n):
        if high[i] != low[i]:
            money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            mfv_arr[i] = money_flow_multiplier * volume[i]

    money_flow_volume = 0.0
    volume_sum = 0.0
    for j in range(length):
        money_flow_volume += mfv_arr[j]
        volume_sum += volume[j]

    if volume_sum != 0:
        cmf[length - 1] = money_flow_volume / volume_sum

    for i in range(length, n):
        money_flow_volume += mfv_arr[i] - mfv_arr[i - length]
        volume_sum += volume[i] - volume[i - length]

        if volume_sum != 0:
            cmf[i] = money_flow_volume / volume_sum

    return cmf

np.random.seed(42)
high = np.random.rand(100) * 100 + 10
low = high - np.random.rand(100) * 5
close = low + np.random.rand(100) * 5
volume = np.random.randint(100, 1000, 100).astype(float)

# Test MFI
mfi_1 = mfi_orig(high, low, close, volume)
mfi_2 = mfi_opt(high, low, close, volume)
np.testing.assert_allclose(mfi_1, mfi_2, equal_nan=True)
print("MFI Match!")

# Test CMF
cmf_1 = cmf_orig(high, low, close, volume, 14)
cmf_2 = cmf_opt(high, low, close, volume, 14)
np.testing.assert_allclose(cmf_1, cmf_2, equal_nan=True)
print("CMF Match!")
