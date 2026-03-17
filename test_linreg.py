import numpy as np

def linreg_numba_orig(close, length=14, r=False):
    n = len(close)
    linreg = np.full(n, np.nan)

    x = np.arange(1, length + 1)
    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    for i in range(length - 1, n):
        series = close[i - length + 1:i + 1]
        y_sum = np.sum(series)
        xy_sum = np.sum(x * series)

        m = (length * xy_sum - x_sum * y_sum) / divisor

        if r:
            y2_sum = np.sum(series * series)
            rn = length * xy_sum - x_sum * y_sum
            rd = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
            linreg[i] = rn / rd
        else:
            linreg[i] = m

    return linreg

def linreg_numba_opt(close, length=14, r=False):
    n = len(close)
    linreg = np.full(n, np.nan)

    x = np.arange(1, length + 1)
    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    if n < length:
        return linreg

    # INITIALIZATION FOR THE FIRST WINDOW
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

    # Calculate for index `length - 1`
    m = (length * xy_sum - x_sum * y_sum) / divisor
    if r:
        rn = length * xy_sum - x_sum * y_sum
        rd = np.sqrt(divisor * (length * y2_sum - y_sum * y_sum))
        linreg[length - 1] = rn / rd
    else:
        linreg[length - 1] = m

    # RUNNING SUM FOR THE REST OF THE ARRAY
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
            # prevent negative values in sqrt due to floating point inaccuracies
            rd_val = divisor * (length * y2_sum - y_sum * y_sum)
            rd = np.sqrt(max(0.0, rd_val))
            linreg[i] = rn / rd if rd != 0 else 0.0
        else:
            linreg[i] = m

    return linreg

close = np.random.rand(100) * 100
res1 = linreg_numba_orig(close, length=14, r=False)
res2 = linreg_numba_opt(close, length=14, r=False)
print("m close?", np.allclose(res1, res2, equal_nan=True))

res3 = linreg_numba_orig(close, length=14, r=True)
res4 = linreg_numba_opt(close, length=14, r=True)
print("r close?", np.allclose(res3, res4, equal_nan=True))
