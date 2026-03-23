# ti_numba – Copilot Instructions

High-performance technical indicators library powered by Numba JIT compilation. Every indicator is a `@njit(cache=True)` pure function with strict O(N) complexity and zero lookahead bias.

## Essential Commands

```bash
# Pre-compile all Numba functions (must run after adding new indicators)
python compile.py

# Run full test suite
pytest tests/

# Run usage/integration examples (requires ccxt for live data)
python examples/usage_examples.py
```

## Architecture

```
src/indicators/<category>/   # Pure @njit(cache=True) functions  →  <name>_numba()
src/base/indicator_categories.py  # OOP category wrappers  →  class MomentumIndicators, etc.
src/base/indicator_base.py        # IndicatorBase dataclass, get_data(), calculate_indicator()
src/base/technical_indicators.py  # TechnicalIndicators façade (user-facing entry point)
tests/base/                        # Integration tests for wrappers
tests/indicators/<category>/       # Mathematical correctness tests per indicator
examples/usage_examples.py         # CCXT + pandas integration reference
compile.py                         # AOT compilation trigger for all @njit functions
.jules/quant.md                    # Agent learning journal – read before touching indicators
```

**Data flow:** `TechnicalIndicators.get_data(ohlcv)` → category wrapper → `@njit` function → `np.ndarray`

**Input format:** `[open, high, low, close, volume]` — 5-column NumPy array, optional leading timestamp column, or a pandas OHLCV DataFrame.

## Indicator Implementation Rules

### Naming
- Raw Numba function: `<indicator>_numba` in `src/indicators/<category>/<category>_indicators.py`
- Category wrapper method: `<indicator>(self, ...)` in `src/base/indicator_categories.py`, calls `self._base.calculate_indicator(func, *args, required_length=...)`
- Register in `src/indicators/<category>/__init__.py` and in `compile.py`

### Performance: always O(N)
- **Never** use `np.sum(arr[i-L:i])`, `np.mean(arr[i-L:i])`, or `np.dot` on a slice inside a loop — this creates O(N·K) bottlenecks that Numba cannot optimize away.
- Use O(N) running sums: `window_sum += new_val - old_val`.
- For WMA: maintain `window_sum` (unweighted) and `window_sum_w` (weighted), update via `window_sum_w += length * val[i] - window_sum`.
- Pre-calculate derived arrays (e.g., absolute diffs, money-flow arrays) in a first O(N) pass, then run the rolling sum in a second pass.

### Lookahead bias — absolute prohibitions
- **Never** use `np.roll` on trading data. `np.roll(close, length)` wraps the newest values into the past.
- **Never** use negative indices in lookback loops. `close[i - length]` becomes `close[-1]` when `i < length`, silently injecting the last price into history. Always guard: `for i in range(length, n)` (not `n - 1`).
- Minimum loop start index must be `>= maximum lookback window`.

### Numerical safety
- Always wrap variance before `sqrt`: `stdev = np.sqrt(max(0.0, variance))`.
- Guard all divisions: `if std_dev == 0: continue` (or early-return).
- For kurtosis guard `length < 4`; for skewness guard `length < 3`.
- **Do not** use algebraic running-sum expansions for 3rd/4th-moment statistics (catastrophic cancellation on real price data). Use a hybrid: O(N) running mean + O(K) slice for deviations, or Welford's algorithm.

### Histogram / binning
- The final bin's upper boundary must be inclusive (`>=`). Using strict `<` for all bins silently drops the maximum value.

## Testing Conventions

- Each new `_numba` function needs a test in `tests/indicators/<category>/` checking:
  1. Output shape matches input length.
  2. First `length - 1` values are `NaN`.
  3. No lookahead: shift the series by one bar and verify results shift accordingly.
  4. Edge cases: all-equal prices, zero volume, `length == len(data)`.
- Use `scipy` reference implementations to assert mathematical correctness where available.

## Key References

- See [.jules/quant.md](.jules/quant.md) for a growing log of discovered bugs and learnings — consult it before modifying any indicator.
- See [README.md](../README.md) for the full indicator catalogue and usage examples.
- See [examples/usage_examples.py](../examples/usage_examples.py) for the canonical CCXT + pandas integration pattern.
