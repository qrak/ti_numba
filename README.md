# High-performance technical indicators using Numba JIT compilation

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Numba](https://img.shields.io/badge/numba-0.60.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Plot Example](https://raw.githubusercontent.com/qrak/ti_numba/master/examples/plot.png)

## Features

- Multiple indicator categories:
    - Momentum Indicators
    - Overlap Studies
    - Price Transform
    - Sentiment Indicators
    - Statistical Functions
    - Support/Resistance Levels
    - Trend Indicators
    - Volatility Indicators
    - Volume Indicators

## Requirements

- Python >= 3.9
- pandas == 2.2.1
- numpy ~= 1.26.4
- numba == 0.60.0
- ccxt == 4.4.28
- matplotlib == 3.8.3

## Usage with numpy data

```python
import numpy as np
from src.ti_numba.base import TechnicalIndicators

indicators = TechnicalIndicators(measure_time=True)

data = np.array([
    [10.0, 12.0, 9.0, 11.0, 1000.0],  # [open, high, low, close, volume]
    [11.0, 13.0, 10.0, 12.0, 1500.0],
    [12.0, 14.0, 11.0, 13.0, 2000.0]
])

indicators.get_data(data)

rsi = indicators.momentum.rsi(length=14)
macd = indicators.momentum.macd()
bb = indicators.volatility.bollinger_bands()
```

## Usage with pandas dataframe data

```python
import pandas as pd
from src.ti_numba.base import TechnicalIndicators

df = pd.DataFrame({
    'Open': [10.0, 11.0, 12.0],
    'High': [12.0, 13.0, 14.0],
    'Low': [9.0, 10.0, 11.0],
    'Close': [11.0, 12.0, 13.0],
    'Volume': [1000.0, 1500.0, 2000.0]
})

indicators = TechnicalIndicators(measure_time=True)
indicators.get_data(df)

stoch = indicators.momentum.stochastic()
adx = indicators.trend.adx()
vwap = indicators.vol.rolling_vwap()
```

## Numpy array usage with datetime.

```python
import numpy as np
from src.ti_numba.base import TechnicalIndicators
from datetime import datetime, timedelta

dates = [datetime.now() + timedelta(days=x) for x in range(3)]
data_with_time = np.array([
    [dates[0].timestamp(), 10.0, 12.0, 9.0, 11.0, 1000.0],
    [dates[1].timestamp(), 11.0, 13.0, 10.0, 12.0, 1500.0],
    [dates[2].timestamp(), 12.0, 14.0, 11.0, 13.0, 2000.0]
])

indicators = TechnicalIndicators(save_to_csv=True)
indicators.get_data(data_with_time)

supertrend = indicators.trend.supertrend()
mfi = indicators.vol.mfi()
```

## Parameters

- measure_time: bool, optional (default=False)
    - If True, prints execution time for each indicator calculation
- save_to_csv: bool, optional (default=False)
    - If True, saves indicator results to CSV files

## Data Input Formats

### 1. NumPy array:
- Basic format: [open, high, low, close, volume]
- With timestamp: [timestamp, open, high, low, close, volume]

### 2. Pandas DataFrame:
- Required columns (case-insensitive):
    - 'open' or 'Open'
    - 'high' or 'High'
    - 'low' or 'Low'
    - 'close' or 'Close'
    - 'volume' or 'Volume'

## Properties

- open: np.ndarray - Open price data
- high: np.ndarray - High price data
- low: np.ndarray - Low price data
- close: np.ndarray - Close price data
- volume: np.ndarray - Volume data

## Notes

- First calculation includes Numba compilation time
- Subsequent calculations are significantly faster due to Numba's caching
- Case-insensitive column matching allows for flexible data input
- Timestamp column in NumPy arrays is automatically detected and handled

### Momentum Indicators

- **RSI (Relative Strength Index):** Measures the speed and change of price movements, typically used to identify overbought or oversold conditions.
- **MACD (Moving Average Convergence Divergence):** A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
- **Stochastic Oscillator:** Compares a particular closing price of a security to a range of its prices over a certain period of time.
- **ROC (Rate of Change):** Measures the percentage change between the most recent price and the price "n" periods in the past.
- **Momentum:** Measures the rate of change of a security's price.
- **Williams %R:** A momentum indicator that measures overbought and oversold levels.
- **TSI (True Strength Index):** A momentum oscillator that ranges between -100 and +100.
- **RMI (Relative Momentum Index):** A variation of RSI that uses momentum instead of price change.
- **PPO (Percentage Price Oscillator):** Similar to MACD, but expressed as a percentage.
- **Coppock Curve:** A long-term price momentum indicator used primarily to identify major bottoms in the stock market.

### Overlap Indicators

- **EMA (Exponential Moving Average):** A type of moving average that places a greater weight and significance on the most recent data points.
- **SMA (Simple Moving Average):** The average of a selected range of prices, usually closing prices, by the number of periods in that range.
- **EWMA (Exponentially Weighted Moving Average):** A moving average that applies weighting factors which decrease exponentially.

### Price Transform Indicators

- **Log Return:** Calculates the logarithmic return of a security over a specified period.
- **Percent Return:** Measures the percentage change in price over a specified period.
- **PDist (Price Distance):** Calculates the distance between the open, high, low, and close prices over a specified drift period.

### Sentiment Indicators

- **Fear and Greed Index:** A composite indicator that measures the market sentiment using various factors like RSI, MACD, and MFI.

### Statistical Indicators

- **Kurtosis:** Measures the "tailedness" of the probability distribution of a real-valued random variable.
- **Skew:** Measures the asymmetry of the probability distribution of a real-valued random variable.
- **Standard Deviation (StDev):** Measures the amount of variation or dispersion of a set of values.
- **Variance:** Measures how far a set of numbers are spread out from their average value.
- **Z-Score:** Measures the number of standard deviations a data point is from the mean.
- **MAD (Mean Absolute Deviation):** Measures the average absolute deviations from a central point.
- **Quantile:** Divides the range of a probability distribution into continuous intervals with equal probabilities.
- **Entropy:** Measures the uncertainty or randomness in a data set.
- **Hurst Exponent:** Measures the long-term memory of time series data.
- **Linear Regression (LinReg):** A linear approach to modeling the relationship between a dependent variable and one or more independent variables.
- **APA Adaptive EOT:** An adaptive version of the Ehlers Early Onset Trend indicator that dynamically adjusts to market conditions. It aims to provide early detection of trend changes by reducing lag and filtering out noise, making it a valuable tool for confirming trend reversals.
- **Calculate EOT:** A trend-following indicator developed by John F. Ehlers, designed to detect the onset of a trend with minimal lag. It uses a combination of the Super Smoother Filter and Roofing Filter to eliminate noise and spectral dilation, providing early signals for trend detection.

### Support and Resistance Indicators

- **Support and Resistance:** Identifies horizontal support and resistance levels based on historical price data.
- **Find Support and Resistance:** Calculates support and resistance levels using advanced methods for more precise identification.
- **Advanced Support and Resistance:** Uses additional parameters like strength threshold and volume factor for enhanced support and resistance detection.
- **Fibonacci Retracement:** Identifies potential reversal levels using Fibonacci ratios.
- **Fibonacci Bollinger Bands:** Combines Fibonacci retracement levels with Bollinger Bands for dynamic support and resistance levels.
- **Floating Levels:** Determines dynamic support and resistance levels based on recent price action and specified parameters.

### Trend Indicators

- **ADX (Average Directional Index):** Measures the strength of a trend, regardless of its direction.
- **Supertrend:** A trend-following indicator that provides buy and sell signals based on price action and volatility.
- **Ichimoku Cloud:** A comprehensive indicator that defines support and resistance, identifies trend direction, gauges momentum, and provides trading signals.
- **Parabolic SAR:** A trend-following indicator that provides potential reversal points.
- **Vortex Indicator:** Identifies the start of a new trend and confirms the direction of an existing trend.
- **TRIX (Triple Exponential Average):** A momentum oscillator that displays the percent rate of change of a triple exponentially smoothed moving average.
- **PFE (Polarized Fractal Efficiency):** Measures the efficiency of price movement, indicating whether a trend is strong or weak.

### Volatility Indicators

- **ATR (Average True Range):** Measures market volatility by decomposing the entire range of an asset price for that period.
- **Bollinger Bands:** Consists of a middle band (SMA) and two outer bands that are standard deviations away from the middle band, indicating volatility.
- **Chandelier Exit:** A volatility-based indicator used to set trailing stop-losses.
- **VHF (Vertical Horizontal Filter):** Determines whether a market is trending or in a trading range.
- **EBSW (Elder's Bull and Bear Power):** Measures the power of bulls and bears in the market.

### Volume Indicators

- **MFI (Money Flow Index):** A momentum indicator that uses price and volume data to identify overbought or oversold conditions.
- **OBV (On-Balance Volume):** Measures buying and selling pressure as a cumulative indicator that adds volume on up days and subtracts volume on down days.
- **PVT (Price Volume Trend):** Combines price and volume to confirm the strength of price trends.
- **Chaikin Money Flow:** Measures the accumulation-distribution line of moving average convergence-divergence.
- **Accumulation Distribution Line:** A volume-based indicator designed to measure the cumulative flow of money into and out of a security.
- **Force Index:** Combines price movement and volume to identify the strength of a trend.
- **EOM (Ease of Movement):** Relates an asset's price change to its volume and is particularly useful for assessing the strength of a trend.
- **Volume Profile:** Displays trading activity over a specified time period at specified price levels.
- **Rolling VWAP (Volume Weighted Average Price):** Provides the average price a security has traded at throughout the day, based on both volume and price.
- **TWAP (Time Weighted Average Price):** A trading algorithm that breaks up a large order and releases smaller portions of the order to the market over a specified time period.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

