import pytest
from unittest.mock import MagicMock
import numpy as np

from src.base.indicator_base import IndicatorBase
from src.base.indicator_categories import OverlapIndicators, MomentumIndicators
from src.indicators.overlap.overlap_indicators import ema_numba, sma_numba, ewma_numba

class TestOverlapIndicators:
    @pytest.fixture
    def mock_base(self):
        base = MagicMock(spec=IndicatorBase)
        # Fix: don't mock on the class MagicMock itself
        base.close = np.array([1.0, 2.0, 3.0])
        return base

    @pytest.fixture
    def overlap_indicators(self, mock_base):
        return OverlapIndicators(mock_base)

    def test_ema(self, mock_base, overlap_indicators):
        data_series = np.array([1.0, 2.0, 3.0])
        length = 10

        overlap_indicators.ema(data_series, length)

        mock_base.calculate_indicator.assert_called_once_with(
            ema_numba,
            data_series,
            length,
            required_length=length
        )

    def test_sma(self, mock_base, overlap_indicators):
        data_series = np.array([1.0, 2.0, 3.0])
        length = 10

        overlap_indicators.sma(data_series, length)

        mock_base.calculate_indicator.assert_called_once_with(
            sma_numba,
            data_series,
            length,
            required_length=length
        )

    def test_ewma(self, mock_base, overlap_indicators):
        span = 10

        overlap_indicators.ewma(span)

        mock_base.calculate_indicator.assert_called_once_with(
            ewma_numba,
            mock_base.close,
            span
        )

class TestMomentumIndicators:
    @pytest.fixture
    def mock_base(self):
        base = MagicMock(spec=IndicatorBase)
        base.close = np.array([1.0, 2.0, 3.0])
        base.high = np.array([1.2, 2.2, 3.2])
        base.low = np.array([0.8, 1.8, 2.8])
        return base

    @pytest.fixture
    def mock_overlap(self):
        overlap = MagicMock(spec=OverlapIndicators)
        overlap.sma.return_value = np.array([1.0, 2.0, 3.0])
        return overlap

    @pytest.fixture
    def momentum_indicators(self, mock_base, mock_overlap):
        return MomentumIndicators(mock_base, mock_overlap)

    def test_rsi(self, mock_base, momentum_indicators):
        from src.indicators.momentum import rsi_numba
        length = 14
        momentum_indicators.rsi(length)
        mock_base.calculate_indicator.assert_called_once_with(
            rsi_numba,
            mock_base.close,
            length,
            required_length=length
        )

    def test_macd(self, mock_base, momentum_indicators):
        from src.indicators.momentum import macd_numba
        fast_length, slow_length, signal_length = 12, 26, 9
        momentum_indicators.macd(fast_length, slow_length, signal_length)
        mock_base.calculate_indicator.assert_called_once_with(
            macd_numba,
            mock_base.close,
            fast_length,
            slow_length,
            signal_length,
            required_length=slow_length
        )

    def test_stochastic(self, mock_base, momentum_indicators):
        from src.indicators.momentum import stochastic_numba
        period_k, smooth_k, period_d = 5, 3, 3
        momentum_indicators.stochastic(period_k, smooth_k, period_d)
        mock_base.calculate_indicator.assert_called_once_with(
            stochastic_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            period_k,
            smooth_k,
            period_d,
            required_length=3
        )

class TestVolumeIndicators:
    @pytest.fixture
    def mock_base(self):
        class DummyBase:
            def __init__(self):
                self.n = 25
                self.close = np.full(self.n, 12.0)
                self.high = np.full(self.n, 12.0)
                self.low = np.full(self.n, 8.0)
                self.volume = np.full(self.n, 1e5)

            def calculate_indicator(self, func, *args, **kwargs):
                return func(*args)

        return DummyBase()

    @pytest.fixture
    def volume_indicators(self, mock_base):
        from src.base.indicator_categories import VolumeIndicators
        return VolumeIndicators(mock_base)

    def test_chaikin_money_flow_mathematics(self, mock_base, volume_indicators):
        # A bullish setup where close is at high, should yield CMF of 1.0 after the window
        length = 5
        result = volume_indicators.chaikin_money_flow(length=length)

        # Verify shape
        assert result.shape == (mock_base.n,)

        # Verify warm-up NaNs
        assert np.isnan(result[:length - 1]).all()

        # Verify actual mathematical output
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0)