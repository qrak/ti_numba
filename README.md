# ti_numba: High-Performance Technical Indicators

High-performance, mathematically rigorous technical indicators library powered by **Numba JIT compilation** for Python. Designed for quantitative developers, algorithmic traders, and data scientists who require lighting-fast indicator computation without lookahead bias.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Numba](https://img.shields.io/badge/numba-0.61.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📑 Table of Contents

- [ti\_numba: High-Performance Technical Indicators](#ti_numba-high-performance-technical-indicators)
  - [📑 Table of Contents](#-table-of-contents)
  - [🚀 Key Features](#-key-features)
  - [🛠 Tech Stack](#-tech-stack)
  - [⚙️ Prerequisites](#️-prerequisites)
  - [📦 Getting Started](#-getting-started)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Pre-Compile Numba Functions (Crucial)](#3-pre-compile-numba-functions-crucial)
  - [🏗 Architecture Overview](#-architecture-overview)
    - [Directory Structure](#directory-structure)
    - [Data Flow](#data-flow)
  - [💻 Usage Examples](#-usage-examples)
    - [1. Using NumPy Arrays (Fastest)](#1-using-numpy-arrays-fastest)
    - [2. Using Pandas DataFrames](#2-using-pandas-dataframes)
    - [3. Handling Timestamps](#3-handling-timestamps)
  - [📈 Available Indicators](#-available-indicators)
    - [Momentum Indicators](#momentum-indicators)
    - [Overlap Indicators](#overlap-indicators)
    - [Price Transform Indicators](#price-transform-indicators)
    - [Sentiment Indicators](#sentiment-indicators)
    - [Statistical Indicators](#statistical-indicators)
    - [Support & Resistance Indicators](#support--resistance-indicators)
    - [Trend Indicators](#trend-indicators)
    - [Volatility Indicators](#volatility-indicators)
    - [Volume Indicators](#volume-indicators)
  - [🧰 Available Scripts](#-available-scripts)
  - [🧪 Testing](#-testing)
    - [Running the Test Suite](#running-the-test-suite)
  - [📜 License](#-license)

---

## 🚀 Key Features

1. **Numba JIT Compilation:** All indicators are heavily optimized with Numba's `@njit(cache=True)`, offering near C-level execution speeds and completely bypassing standard pandas calculation bottlenecks.
2. **Zero Lookahead Bias:** Calculations are strictly mathematically guarded and rigorously tested to prevent any leakage of future data into historical calculations.
3. **Comprehensive Coverage:** Over 50+ implemented formulas covering volume, sentiment, trend, statistical models, mathematical transforms, and momentum.
4. **Flexible Input Handling:** First-class support for raw `numpy.ndarray` operations (ultra-fast) or structured `pandas.DataFrame` inputs (convenient).

---

## 🛠 Tech Stack

- **Language:** Python 3.9+
- **JIT Compiler:** Numba 0.61.0
- **Data Manipulation:** NumPy 2.1.3 & Pandas 2.2.3
- **Visualization:** Matplotlib 3.10.0 (Examples only)
- **Testing:** Pytest 9.0.2

---

## ⚙️ Prerequisites

To run this library locally, ensure you have the following installed:

- Python 3.9 or higher
- `pip` or `uv` package manager
- (Optional but recommended) A virtual environment

---

## 📦 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/qrak/ti_numba.git
cd ti_numba
```

### 2. Install Dependencies

You can install all required packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Pre-Compile Numba Functions (Crucial)

**Warning:** The first time Numba runs a decorated function, it must compile it to machine code. This creates a one-time cold-start latency. We provide a compilation script that pre-triggers all functions and caches the operations locally.

```bash
# Run the compiler to build the Numba cache
python compile.py
```


---

## 🏗 Architecture Overview

### Directory Structure

```text
├── examples/               # Usage scripts and CCXT data examples
│   └── usage_examples.py   # Benchmark tests comparing Pandas vs Numpy
├── src/                    # Main source code
│   ├── base/               # Indicator categories and class wrappers
│   │   ├── indicator_categories.py
│   │   └── technical_indicators.py
│   └── indicators/         # Raw Numba mathematical implementations
│       ├── momentum/
│       ├── overlap/
│       ├── price/
│       ├── sentiment/
│       ├── statistical/
│       ├── support_resistance/
│       ├── trend/
│       ├── volatility/
│       └── volume/
├── tests/                  # Pytest directory (250+ assertions)
│   ├── base/               # Integration & Wrapper testing
│   └── indicators/         # Category mathematical assertions
├── .jules/                 # Quantitative engineering journals & logs
├── compile.py              # System JIT compilation script
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

### Data Flow

```text
User Input (Array Matrix/Pandas DataFrame) 
  → TechnicalIndicators (Sanitizer) 
    → Category Wrappers (e.g. MomentumIndicators)
      → Core @njit Numpy Implementations (e.g. src/indicators/momentum/...)
        → Output Numpy Array
```

---

## 💻 Usage Examples

The `TechnicalIndicators` class manages the data state and delegates calls to the distinct categories.

### 1. Using NumPy Arrays (Fastest)

```python
import numpy as np
from src.base import TechnicalIndicators

# initialize with debugging attributes if required
indicators = TechnicalIndicators(measure_time=True)

# Format: [open, high, low, close, volume]
data = np.array([
    [10.0, 12.0, 9.0, 11.0, 1000.0],  
    [11.0, 13.0, 10.0, 12.0, 1500.0],
    [12.0, 14.0, 11.0, 13.0, 2000.0]
])

indicators.get_data(data)

# Call indicator categories directly
rsi = indicators.momentum.rsi(length=14)
macd = indicators.momentum.macd()
bb = indicators.volatility.bollinger_bands()

print(rsi)
```

### 2. Using Pandas DataFrames

For convenience, you can pass a standard OHLCV dataframe:

```python
import pandas as pd
from src.base import TechnicalIndicators

df = pd.DataFrame({
    'Open': [10.0, 11.0, 12.0],
    'High': [12.0, 13.0, 14.0],
    'Low': [9.0, 10.0, 11.0],
    'Close': [11.0, 12.0, 13.0],
    'Volume': [1000.0, 1500.0, 2000.0]
})

indicators = TechnicalIndicators()
indicators.get_data(df)

stoch = indicators.momentum.stochastic()
adx = indicators.trend.adx()
```

### 3. Handling Timestamps

```python
import numpy as np
from datetime import datetime, timedelta
from src.base import TechnicalIndicators

dates = [datetime.now() + timedelta(days=x) for x in range(3)]
# Format: [timestamp, open, high, low, close, volume]
data_with_time = np.array([
    [dates[0].timestamp(), 10.0, 12.0, 9.0, 11.0, 1000.0],
    [dates[1].timestamp(), 11.0, 13.0, 10.0, 12.0, 1500.0]
])

# Turn on CSV dumps if you wish to analyze the output visually
indicators = TechnicalIndicators(save_to_csv=True) 
indicators.get_data(data_with_time)

mfi = indicators.vol.mfi()
```

---

## 📈 Available Indicators

### Momentum Indicators
- **RSI (Relative Strength Index):** Measures speed/change of price movements.
- **MACD (Moving Average Convergence Divergence):** Relationship between two moving averages.
- **Stochastic Oscillator:** Compares a closing price to its range.
- **ROC (Rate of Change):** Percentage change over `n` periods.
- **Momentum:** Raw momentum value.
- **Williams %R:** Overbought/oversold levels.
- **TSI (True Strength Index):** Ranges `-100` to `+100`.
- **RMI (Relative Momentum Index):** Uses momentum instead of typical price change.
- **PPO (Percentage Price Oscillator):** MACD, but formatted as a percentage.
- **Coppock Curve:** Long-term momentum identifier.
- **Detect RSI Divergence:** Custom divergence checks.
- **Relative Strength Index:** Benchmark relative comparative strength.
- **KST (Know Sure Thing):** Smoothed ROC over four frames.
- **UO (Ultimate Oscillator):** Captures multi-timeframe pressure.

### Overlap Indicators
- **EMA (Exponential Moving Average):** Heavy weighting on recent data.
- **SMA (Simple Moving Average):** Flat arithmetic average.
- **EWMA (Exponentially Weighted Moving Average):** Continuous decay exponential average.

### Price Transform Indicators
- **Log Return:** Standard logarithmic compounding returns.
- **Percent Return:** Cumulative percentage return shift.
- **PDist (Price Distance):** Differential span between structural price marks.

### Sentiment Indicators
- **Fear and Greed Index:** Normalised composite oscillator (`0-100`) blending RSI, MACD, and MFI for extreme sentiment bias reads.

### Statistical Indicators
- **Kurtosis / Skew:** Mathematical shape identifiers.
- **Standard Deviation / Variance:** Statistical distribution baselines.
- **Z-Score:** Standard deviation offset from mean.
- **MAD (Mean Absolute Deviation):** Center point variance.
- **Quantile:** Percentage boundaries.
- **Entropy:** Stochastic unpredictability metrics.
- **Hurst Exponent:** Advanced long-term memory analysis.
- **Linear Regression (LinReg):** Algorithmic geometric curves.
- **APA Adaptive EOT / EOT:** Ehlers dynamic/trend onset tools.

### Support & Resistance Indicators
- **Support and Resistance:** High/Low mapping boundaries.
- **Find Support and Resistance:** Proximity detection.
- **Advanced Support and Resistance:** Volumes and threshold thresholds detection.
- **Fibonacci Retracement:** Dynamic mathematical retraction targets.
- **Fibonacci Bollinger Bands:** Cross-calculated Gaussian Fibonacci levels.
- **Floating Levels:** Adaptive local zones.

### Trend Indicators
- **ADX (Average Directional Index):** Pure directionless slope strength.
- **Supertrend:** Price/Volatility directional flip signal.
- **Ichimoku Cloud:** Complex multi-dimensional support mapping.
- **Parabolic SAR:** Stop and reverse targets.
- **Vortex Indicator:** Directional crossing thresholds.
- **TRIX (Triple EMA):** ROC of triple smoothed signals.
- **PFE (Polarized Fractal Efficiency):** Trajectory directional impact rating.

### Volatility Indicators
- **ATR (Average True Range):** Total average step-change range mapping.
- **Bollinger Bands:** SMA bounded by StD ranges.
- **Chandelier Exit:** Trailing extreme dynamic exits.
- **VHF (Vertical Horizontal Filter):** Trend vs range evaluator.
- **EBSW (Elder's Bull and Bear Power):** Pressure analysis.

### Volume Indicators
- **MFI (Money Flow Index):** Volume-weighted RSI (`0-100`).
- **OBV (On-Balance Volume):** Total additive volume tracking.
- **PVT (Price Volume Trend):** ROC correlated volume tracker.
- **Chaikin Money Flow:** A/D bound tracker.
- **A/D Line:** Absolute accumulation.
- **Force Index:** Magnitude volume scale evaluation.
- **EOM (Ease of Movement):** Price range vs proportional volume.
- **Volume Profile:** Density segmentation histograms bounded by target intervals.
- **Rolling VWAP:** Volume-weighted step pricing.
- **TWAP:** Standard algorithmic temporal execution slices.
- **Average Quote Volume:** Scaled transaction magnitude.

---

## 🧰 Available Scripts

| Script | Command | Description |
|---|---|---|
| Compile Caches | `python compile.py` | Triggers Numba initialization on all numerical algorithms, building byte-code for subsequent operations |
| Execution Benchmarks | `python examples/usage_examples.py` | Bootstraps CCXT, loads sample financial logic, and tests Array Vs. DataFrame runtime speeds internally |

---

## 🧪 Testing

The library uses `pytest` and maintains over **250 distinct unit and integration assertions** checking parameter behavior, boundary logic, lack of lookahead bias, NaN handling, div-by-zero protections, and perfect mathematical precision.

### Running the Test Suite

```bash
# Run all tests natively 
python -m pytest tests/ -v

# Run with short tracebacks
python -m pytest tests/ --tb=short
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
