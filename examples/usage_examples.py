import timeit
from typing import Dict, Optional, Tuple, Union

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ti_numba.base import TechnicalIndicators


class MarketDataFetcher:
    def __init__(self) -> None:
        self.exchanges = {
            'binance': ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True
            }),
            'binanceus': ccxt.binanceus({
                'rateLimit': 1200,
                'enableRateLimit': True
            }),
            'kraken': ccxt.kraken({
                'rateLimit': 3000,
                'enableRateLimit': True
            }),
            'kucoin': ccxt.kucoin({
                'rateLimit': 1500,
                'enableRateLimit': True
            }),
            'okx': ccxt.okx({
                'rateLimit': 1000,
                'enableRateLimit': True
            })
        }

        self.exchange_order = ['binance', 'binanceus', 'kraken', 'kucoin', 'okx']

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        for exchange_id in self.exchange_order:
            exchange = self.exchanges[exchange_id]
            try:
                if not exchange.has['fetchOHLCV']:
                    continue

                exchange.load_markets()
                if symbol not in exchange.markets:
                    print(f"{symbol} not found in {exchange_id}, trying next exchange...")
                    continue

                timestamp = exchange.parse8601(since)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, timestamp, limit)

                if not ohlcv:
                    continue

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"Successfully fetched data from {exchange_id}")

                return df

            except Exception as e:
                print(f"Error fetching from {exchange_id}: {str(e)}")
                continue

        print("Failed to fetch data from all exchanges")
        return None


class IndicatorAnalyzer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.indicators = TechnicalIndicators(measure_time=True)
        self.numpy_data = self._prepare_numpy_data()

    def _prepare_numpy_data(self) -> np.ndarray:
        return np.column_stack((
            self.data['open'].values,
            self.data['high'].values,
            self.data['low'].values,
            self.data['close'].values,
            self.data['volume'].values
        ))

    def calculate_indicators(self) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        self.indicators.get_data(self.numpy_data)
        close_array = self.numpy_data[:, 3]

        return {
            'RSI': self.indicators.momentum.rsi(),
            'MACD': self.indicators.momentum.macd(),
            'EMA': self.indicators.overlap.ema(close_array),
            'BB': self.indicators.volatility.bollinger_bands(),
            'ADX': self.indicators.trend.adx()
        }

    def plot_indicators(self, indicators_data: Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Price and BB
        axes[0].plot(self.data['timestamp'], self.data['close'], label='Price')
        if isinstance(indicators_data['BB'], tuple):
            axes[0].plot(self.data['timestamp'], indicators_data['BB'][0], 'r--', label='BB Upper')
            axes[0].plot(self.data['timestamp'], indicators_data['BB'][1], 'g--', label='BB Middle')
            axes[0].plot(self.data['timestamp'], indicators_data['BB'][2], 'r--', label='BB Lower')
        axes[0].set_title('Price and Bollinger Bands')
        axes[0].legend()

        # RSI and MACD
        axes[1].plot(self.data['timestamp'], indicators_data['RSI'], label='RSI')
        axes[1].axhline(y=70, color='r', linestyle='--')
        axes[1].axhline(y=30, color='g', linestyle='--')
        axes[1].set_title('RSI')
        axes[1].legend()

        # MACD
        if isinstance(indicators_data['MACD'], tuple):
            axes[2].plot(self.data['timestamp'], indicators_data['MACD'][0], label='MACD')
            axes[2].plot(self.data['timestamp'], indicators_data['MACD'][1], label='Signal')
            axes[2].bar(self.data['timestamp'], indicators_data['MACD'][2], label='Histogram')
        axes[2].set_title('MACD')
        axes[2].legend()

        plt.tight_layout()
        plt.show()


class DataHandlingBenchmark:
    def __init__(self) -> None:
        self.sample_data_small = self._generate_sample_data(1000)
        self.sample_data_medium = self._generate_sample_data(10000)
        self.sample_data_large = self._generate_sample_data(100000)

    def _generate_sample_data(self, size: int) -> np.ndarray:
        return np.column_stack((
            np.random.uniform(1, 100, size),  # open
            np.random.uniform(1, 100, size),  # high
            np.random.uniform(1, 100, size),  # low
            np.random.uniform(1, 100, size),  # close
            np.random.uniform(1000, 10000, size)  # volume
        ))

    def benchmark_numpy_handling(self, data: np.ndarray) -> float:
        start_time = timeit.default_timer()
        indicators = TechnicalIndicators()
        indicators.get_data(data)
        _ = indicators.trend.adx()
        return timeit.default_timer() - start_time

    def benchmark_list_handling(self, data: np.ndarray) -> float:
        start_time = timeit.default_timer()
        data_list = data.tolist()
        indicators = TechnicalIndicators()
        indicators.get_data(data_list)
        _ = indicators.trend.adx()
        return timeit.default_timer() - start_time

    def benchmark_dataframe_handling(self, data: np.ndarray) -> float:
        start_time = timeit.default_timer()
        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        indicators = TechnicalIndicators()
        indicators.get_data(df)
        _ = indicators.trend.adx()
        return timeit.default_timer() - start_time

    def run_benchmarks(self) -> None:
        datasets = {
            'Small (1K rows)': self.sample_data_small,
            'Medium (10K rows)': self.sample_data_medium,
            'Large (100K rows)': self.sample_data_large
        }

        for dataset_name, dataset in datasets.items():
            print(f"\nBenchmarking {dataset_name}:")
            numpy_time = self.benchmark_numpy_handling(dataset)
            list_time = self.benchmark_list_handling(dataset)
            df_time = self.benchmark_dataframe_handling(dataset)

            print(f"NumPy handling time: {numpy_time:.4f} seconds")
            print(f"List handling time: {list_time:.4f} seconds")
            print(f"DataFrame handling time: {df_time:.4f} seconds")



def main() -> None:
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_ohlcv(
        symbol='BTC/USDT',
        timeframe='1d',
        since='2024-01-01 00:00:00',
        limit=1000
    )

    if df is not None:
        analyzer = IndicatorAnalyzer(df)
        indicators_data = analyzer.calculate_indicators()
        analyzer.plot_indicators(indicators_data)

    benchmark = DataHandlingBenchmark()
    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
