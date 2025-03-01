import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import Dict, Any
import numpy as np
import timeit
from datetime import datetime

from ti_numba.base import IndicatorBase, MomentumIndicators, OverlapIndicators, PriceTransformIndicators, \
    SentimentIndicators, StatisticalIndicators, SupportResistanceIndicators, TrendIndicators, VolatilityIndicators, \
    VolumeIndicators


class DataGenerator:
    def __init__(self, size: int = 1000):
        self.size = size
        self.data = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        close = np.random.random(self.size) * 100 + 50
        high = close + np.random.random(self.size) * 10
        low = close - np.random.random(self.size) * 10
        open_price = close + np.random.random(self.size) * 5 - 2.5
        volume = np.random.random(self.size) * 1000000

        return np.column_stack((
            open_price,
            high,
            low,
            close,
            volume
        ))



class IndicatorCompiler:
    def __init__(self, data: DataGenerator):
        self.data = data
        self.base = IndicatorBase()
        self.base.get_data(data.data)
        self.indicators = self._initialize_indicators()

    def _initialize_indicators(self) -> Dict[str, Any]:
        overlap = OverlapIndicators(self.base)
        return {
            "overlap": overlap,
            "momentum": MomentumIndicators(self.base, overlap),
            "price_transform": PriceTransformIndicators(self.base),
            "sentiment": SentimentIndicators(self.base),
            "statistical": StatisticalIndicators(self.base),
            "support_resistance": SupportResistanceIndicators(self.base),
            "trend": TrendIndicators(self.base),
            "volatility": VolatilityIndicators(self.base),
            "volume": VolumeIndicators(self.base)
        }


    def compile_momentum(self) -> None:
        ind = self.indicators["momentum"]
        ind.rsi()
        ind.macd()
        ind.stochastic()
        ind.roc()
        ind.momentum()
        ind.williams_r()
        ind.tsi()
        ind.rmi()
        ind.ppo()
        ind.coppock_curve()

    def compile_overlap(self) -> None:
        ind = self.indicators["overlap"]
        ind.ema(ind.close)
        ind.sma(ind.close)
        ind.ewma()

    def compile_price_transform(self) -> None:
        ind = self.indicators["price_transform"]
        ind.log_return()
        ind.percent_return()
        ind.pdist()

    def compile_sentiment(self) -> None:
        self.indicators["sentiment"].fear_and_greed_index()

    def compile_statistical(self) -> None:
        ind = self.indicators["statistical"]
        ind.kurtosis()
        ind.skew()
        ind.stdev()
        ind.variance()
        ind.zscore()
        ind.mad()
        ind.quantile()
        ind.entropy()
        ind.hurst()
        ind.linreg()
        ind.apa_adaptive_eot()
        ind.calculate_eot()

    def compile_support_resistance(self) -> None:
        ind = self.indicators["support_resistance"]
        ind.support_resistance()
        ind.find_support_resistance()
        ind.support_resistance_advanced()
        ind.advanced_support_resistance()
        ind.fibonacci_retracement()
        ind.fibonacci_bollinger_bands()
        ind.floating_levels()

    def compile_trend(self) -> None:
        ind = self.indicators["trend"]
        ind.adx()
        ind.supertrend()
        ind.ichimoku_cloud()
        ind.parabolic_sar()
        ind.vortex_indicator()
        ind.trix()
        ind.pfe()

    def compile_volatility(self) -> None:
        ind = self.indicators["volatility"]
        ind.atr()
        ind.bollinger_bands()
        ind.chandelier_exit()
        ind.vhf()
        ind.ebsw()

    def compile_volume(self) -> None:
        ind = self.indicators["volume"]
        ind.mfi()
        ind.obv()
        ind.pvt()
        ind.chaikin_money_flow()
        ind.accumulation_distribution_line()
        ind.force_index()
        ind.eom()
        ind.volume_profile()
        ind.rolling_vwap()
        ind.twap()

    def compile_all(self) -> None:
        compilation_methods = [
            self.compile_momentum,
            self.compile_overlap,
            self.compile_price_transform,
            self.compile_sentiment,
            self.compile_statistical,
            self.compile_support_resistance,
            self.compile_trend,
            self.compile_volatility,
            self.compile_volume
        ]

        for method in compilation_methods:
            method_name = method.__name__.replace('compile_', '')
            start_time = timeit.default_timer()
            method()
            end_time = timeit.default_timer()
            print(f"{method_name:<20} compilation time: {end_time - start_time:.4f} seconds")


def main() -> None:
    print(f"Starting compilation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    start_total = timeit.default_timer()

    data = DataGenerator()
    compiler = IndicatorCompiler(data)
    compiler.compile_all()

    end_total = timeit.default_timer()
    print("-" * 50)
    print(f"Total compilation time: {end_total - start_total:.4f} seconds")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
