## Quant Agent Instructions

To verify lookahead bias, edge cases, and performance regressions:
1. Run `python compile.py` to benchmark compilation time and execution.
2. Run `python examples/usage_examples.py` to verify ccxt/pandas integration.
3. Run `pytest tests/` to verify mathematical correctness and lack of lookahead bias using the testing framework.

**Learning entries** should be added below this line as per instructions:
## 2026-03-05 - [Lookahead Bias via np.roll] **Learning:** `np.roll` inherently applies a circular shift, meaning `np.roll(close, length)` causes the earliest elements of the array to reference the very end of the array, introducing severe future data leakage into historical calculations. It also returns a new array and does not modify in-place, leading to logical bugs if unassigned. **Action:** Never use `np.roll` for sequential trading data; replace it with explicit array slicing (e.g., `close[i-length+1 : i+1]`) and strict loop bounds.

## 2026-03-05 - [Algorithm Error in volume_profile_numba] **Learning:** `volume_profile_numba` uses `np.linspace` for bin edges but applies a strictly less-than (`<`) constraint on the upper bound for *all* bins, even the final one. Because the max value produced by `linspace` equals exactly the local maximum of `close`, the bar corresponding to `max(close)` always drops out of the binning completely, silently discarding volume data. **Action:** When utilizing histograms or volume profiling functions via array bins, strictly verify boundary logic (specifically `>=` vs `<=`); the absolute max boundary bin must encompass the absolute maximum to prevent volume leakage.
