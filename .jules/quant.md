## Quant Agent Instructions

To verify lookahead bias, edge cases, and performance regressions:
1. Run `python compile.py` to benchmark compilation time and execution.
2. Run `python examples/usage_examples.py` to verify ccxt/pandas integration.
3. Run `pytest tests/` to verify mathematical correctness and lack of lookahead bias using the testing framework.

**Learning entries** should be added below this line as per instructions:
## 2026-03-05 - [Lookahead Bias via np.roll] **Learning:** `np.roll` inherently applies a circular shift, meaning `np.roll(close, length)` causes the earliest elements of the array to reference the very end of the array, introducing severe future data leakage into historical calculations. It also returns a new array and does not modify in-place, leading to logical bugs if unassigned. **Action:** Never use `np.roll` for sequential trading data; replace it with explicit array slicing (e.g., `close[i-length+1 : i+1]`) and strict loop bounds.

## 2026-03-05 - [Algorithm Error in volume_profile_numba] **Learning:** `volume_profile_numba` uses `np.linspace` for bin edges but applies a strictly less-than (`<`) constraint on the upper bound for *all* bins, even the final one. Because the max value produced by `linspace` equals exactly the local maximum of `close`, the bar corresponding to `max(close)` always drops out of the binning completely, silently discarding volume data. **Action:** When utilizing histograms or volume profiling functions via array bins, strictly verify boundary logic (specifically `>=` vs `<=`); the absolute max boundary bin must encompass the absolute maximum to prevent volume leakage.
## 2026-03-05 - [Skewness & Kurtosis ZeroDivisionError & Formula Corrections]
**Learning:** `skew_numba` and `kurtosis_numba` contained a fatal `ZeroDivisionError` vulnerability when `std_dev` was 0, and threw `ZeroDivisionError` when standard Pandas formulas yielded divisors like `(length-1)*(length-2)*(length-3)` with small windows (`length < 4` for kurtosis, `length < 3` for skewness). Additionally, they incorrectly used standard deviations calculated with 0 degrees of freedom, rather than `ddof=1` sample standard deviations used by pandas and scipy. `entropy_numba` calculated rolling sum with a $O(N \cdot L)$ lookup instead of $O(N)$ running sum.
**Action:** Guard any division by zero using `if std_dev == 0: continue` and `if length < 4: return ...`. Calculate standard deviations within Numba statistical loops using `ddof=1`. Use running sums for properties like `entropy` which rely on rolling totals.

## 2026-03-07 - [O(N) Optimization in Volume Indicators]
**Learning:** Found multiple instances where volume indicators (`eom_numba`, `twap_numba`, `average_quote_volume_numba`) were recalculating `np.mean` and `np.sum` on rolling array slices inside the main calculation loop, causing an O(N*K) algorithmic bottleneck.
**Action:** Replaced these inner slice calculations with O(1) rolling sum variables (e.g. `sum += new_val - old_val`). This reduces the overall time complexity of the indicator functions to strictly O(N), maximizing Numba loop execution speeds.
## 2024-03-08 - Negative Indexing Lookahead Bias in PFE
**Learning:** In Python and Numba, negative indices correctly wrap around to the end of an array without throwing an IndexError. In indicator loops (like `pfe_numba`), if the loop starts too early (e.g. `i = n - 1`) and uses a lookback calculation like `close[i - n]`, this evaluates to `close[-1]`. This quietly injects the *most recent* data point into historical calculations, introducing severe, silent lookahead bias. Additionally, nested loops calculating rolling sums of squared differences (e.g., in PFE) create an O(N*K) bottleneck that can be cleanly refactored into a fast O(N) running sum of pre-calculated squares.
**Action:** Always verify that loop start indices (`i`) are strictly greater than or equal to the maximum lookback window (`n`) to prevent `i - n` from becoming negative. Convert nested lookback window summations into O(N) running sums using pre-calculated arrays for significant performance gains.
## 2025-02-28 - Vortex Indicator O(N*K) Slice Bottleneck Reduced
**Learning:** Found an O(N*K) algorithmic inefficiency in `vortex_indicator_numba` (`src/indicators/trend/trend_indicators.py`) where `np.sum(tr[i - length + 1:i + 1])` (and related slices) was recalculated on every iteration inside a rolling window loop.
**Action:** Replaced the array slicing with an O(N) running sum, maintaining the same window logic. A guard clause (`if n < length:`) was added to prevent out-of-bounds initialization on the running sum state.
