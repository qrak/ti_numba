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

def vhf_final_fixed2(close, length=28, drift=1):
    n = len(close)
    if n < length:
        return np.full(n, np.nan)

    vhf = np.full(n, np.nan)

    diffs = np.zeros(n)
    for i in range(drift, n):
        diffs[i] = np.abs(close[i] - close[i - drift])

    start_idx = length - 1 + drift

    # We maintain a running sum for each phase of the drift
    # There are `drift` independent running sums.
    running_sums = np.zeros(drift)

    # A single window slice consists of:
    # close[i - length + 1 + drift : i + 1 : drift]
    # Length of this sliced_close array is: ceil(length / drift)
    # Number of differences = length(sliced_close) - 1
    # Note that `length - 1` elements is exactly `ceil(length / drift) - 1` diffs.

    for k in range(drift, start_idx):
        phase = k % drift
        running_sums[phase] += diffs[k]

        # Determine the number of steps that fit in the window for this phase.
        # Let's say we are evaluating `vhf` at index `i`.
        # The earliest index included in the difference is `i - length + 1 + drift`
        # old_k must be the index that was dropped from the window calculation.

    for i in range(start_idx, n):
        hcp = np.max(close[i - length + 1:i + 1:drift])
        lcp = np.min(close[i - length + 1:i + 1:drift])

        # Calculate sum_diff using a direct loop for this exact window
        # To avoid the complexity of rolling sums with variable step sizes that don't evenly divide
        sum_diff = 0.0
        for k in range(i - length + 1 + drift, i + 1, drift):
            sum_diff += diffs[k]

        if sum_diff != 0:
            vhf[i] = np.abs(hcp - lcp) / sum_diff
        else:
            vhf[i] = 0

    return vhf


vhf_orig_jit = njit(vhf_original)
vhf_final_jit = njit(vhf_final_fixed2)

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
