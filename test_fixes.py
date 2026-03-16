import numpy as np
from numba import njit
import time

def vhf_original(close, length=28, drift=1):
    n = len(close)
    if n < length:
        return np.full(n, np.nan)

    vhf = np.full(n, np.nan)

    for i in range(length - 1 + drift, n):
        hcp = np.max(close[i - length + 1:i + 1:drift])
        lcp = np.min(close[i - length + 1:i + 1:drift])

        sliced_close = close[i - length + 1:i + 1:drift]

        # Manually compute the differences
        diff = np.abs(sliced_close[1:] - sliced_close[:-1])
        sum_diff = np.sum(diff)

        # Handle division by zero
        if sum_diff != 0:
            vhf[i] = np.abs(hcp - lcp) / sum_diff
        else:
            vhf[i] = 0

    return vhf

def vhf_final_fixed(close, length=28, drift=1):
    n = len(close)
    if n < length:
        return np.full(n, np.nan)

    vhf = np.full(n, np.nan)

    # Precompute absolute differences with the given drift
    # diffs[i] will store abs(close[i] - close[i - drift])
    diffs = np.zeros(n)
    for i in range(drift, n):
        diffs[i] = np.abs(close[i] - close[i - drift])

    start_idx = length - 1 + drift

    # We maintain a running sum for each phase of the drift
    # There are `drift` independent running sums.
    running_sums = np.zeros(drift)

    # Number of elements in the slice close[i - length + 1 : i + 1 : drift]
    # The indices are i, i - drift, i - 2*drift, ..., down to >= i - length + 1
    # Number of steps = floor((length - 1) / drift)
    # The number of elements is num_steps + 1
    # The number of differences is num_steps
    num_diffs = (length - 1) // drift

    # Initialize the running sums up to start_idx - 1
    for k in range(drift, start_idx):
        phase = k % drift
        running_sums[phase] += diffs[k]

        # Remove elements that fall out of the window
        old_k = k - num_diffs * drift
        if old_k >= drift:
            running_sums[phase] -= diffs[old_k]

    for i in range(start_idx, n):
        hcp = np.max(close[i - length + 1:i + 1:drift])
        lcp = np.min(close[i - length + 1:i + 1:drift])

        phase = i % drift
        running_sums[phase] += diffs[i]

        old_k = i - num_diffs * drift
        if old_k >= drift:
            running_sums[phase] -= diffs[old_k]

        sum_diff = running_sums[phase]

        if sum_diff != 0:
            vhf[i] = np.abs(hcp - lcp) / sum_diff
        else:
            vhf[i] = 0

    return vhf


def _mfi_window(window_high, window_low, window_close, window_volume, mfi_length, window_size):
    mfi_list = np.full(window_size, np.nan)
    tp = (window_high + window_low + window_close) / 3
    rmf = tp * window_volume

    if window_size <= mfi_length:
        return mfi_list

    # FIXED: Converted O(N*K) slice sums to O(N) running sums
    pmf_arr = np.zeros(window_size)
    nmf_arr = np.zeros(window_size)

    for i in range(1, window_size):
        if tp[i] > tp[i - 1]:
            pmf_arr[i] = rmf[i]
        elif tp[i] < tp[i - 1]:
            nmf_arr[i] = rmf[i]

    pmf = 0.0
    nmf = 0.0
    for i in range(1, mfi_length + 1):
        pmf += pmf_arr[i]
        nmf += nmf_arr[i]

    if nmf == 0:
        mfi_list[mfi_length] = 100
    else:
        mfr = pmf / nmf
        mfi_list[mfi_length] = 100 * mfr / (1 + mfr)

    for i in range(mfi_length + 1, window_size):
        pmf += pmf_arr[i] - pmf_arr[i - mfi_length]
        nmf += nmf_arr[i] - nmf_arr[i - mfi_length]

        if nmf == 0:
            mfi_list[i] = 100
        else:
            mfr = pmf / nmf
            mfi_list[i] = 100 * mfr / (1 + mfr)

    return mfi_list

vhf_orig_jit = njit(vhf_original)
vhf_final_jit = njit(vhf_final_fixed)

_ = vhf_orig_jit(np.random.random(100), 28, 1)
_ = vhf_final_jit(np.random.random(100), 28, 1)

for test_drift in [1, 2, 5]:
    for test_length in [10, 28, 50]:
        print(f"\nTesting length={test_length}, drift={test_drift}")
        close = np.random.random(1000)
        orig = vhf_orig_jit(close, test_length, test_drift)
        final = vhf_final_jit(close, test_length, test_drift)
        try:
            np.testing.assert_allclose(orig, final, equal_nan=True)
            print("MATCH")
        except AssertionError as e:
            print("MISMATCH", e)

# test mfi bounds
print("\nTesting MFI Bounds")
high = np.random.random(10)
low = np.random.random(10)
close = np.random.random(10)
vol = np.random.random(10)
res = _mfi_window(high, low, close, vol, 14, 10)
print(res)
