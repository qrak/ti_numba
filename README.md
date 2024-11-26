# High-performance technical indicators using Numba JIT compilation

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Numba](https://img.shields.io/badge/numba-0.60.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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

## Usage with data with time

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
