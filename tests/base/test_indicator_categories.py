import pytest
from unittest.mock import MagicMock
import numpy as np

import pytest
from src.base.indicator_base import IndicatorBase
from src.base.indicator_categories import OverlapIndicators, MomentumIndicators, VolumeIndicators
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
        base = MagicMock(spec=IndicatorBase)
        base.close = np.array([1.0, 2.0, 3.0])
        base.high = np.array([1.2, 2.2, 3.2])
        base.low = np.array([0.8, 1.8, 2.8])
        base.volume = np.array([100.0, 200.0, 150.0])
        return base

    @pytest.fixture
    def volume_indicators(self, mock_base):
        return VolumeIndicators(mock_base)

    def test_cci(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import cci_numba
        length, c = 14, 0.015
        volume_indicators.cci(length, c)
        mock_base.calculate_indicator.assert_called_once_with(
            cci_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            length,
            c,
            required_length=length
        )

    def test_mfi(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import mfi_numba
        length, drift = 14, 1
        volume_indicators.mfi(length, drift)
        mock_base.calculate_indicator.assert_called_once_with(
            mfi_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            mock_base.volume,
            length,
            drift,
            required_length=length
        )

    def test_obv(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import obv_numba
        length, initial = 14, 1
        volume_indicators.obv(length, initial)
        mock_base.calculate_indicator.assert_called_once_with(
            obv_numba,
            mock_base.close,
            mock_base.volume,
            length,
            initial,
            required_length=length
        )

    def test_pvt(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import pvt_numba
        length, drift = 14, 1
        volume_indicators.pvt(length, drift)
        mock_base.calculate_indicator.assert_called_once_with(
            pvt_numba,
            mock_base.close,
            mock_base.volume,
            length,
            drift,
            required_length=length
        )

    def test_chaikin_money_flow(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import chaikin_money_flow_numba
        length = 20
        volume_indicators.chaikin_money_flow(length)
        mock_base.calculate_indicator.assert_called_once_with(
            chaikin_money_flow_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            mock_base.volume,
            length,
            required_length=length
        )

    def test_accumulation_distribution_line(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import ad_line_numba
        volume_indicators.accumulation_distribution_line()
        mock_base.calculate_indicator.assert_called_once_with(
            ad_line_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            mock_base.volume
        )

    def test_force_index(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import force_index_numba
        length = 13
        volume_indicators.force_index(length)
        mock_base.calculate_indicator.assert_called_once_with(
            force_index_numba,
            mock_base.close,
            mock_base.volume,
            length,
            required_length=length + 1
        )

    def test_eom(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import eom_numba
        length, divisor, drift = 14, 100000000, 1
        volume_indicators.eom(length, divisor, drift)
        mock_base.calculate_indicator.assert_called_once_with(
            eom_numba,
            mock_base.high,
            mock_base.low,
            mock_base.volume,
            length,
            divisor,
            drift,
            required_length=length
        )

    def test_volume_profile(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import volume_profile_numba
        length, num_bins = 48, 10
        volume_indicators.volume_profile(length, num_bins)
        mock_base.calculate_indicator.assert_called_once_with(
            volume_profile_numba,
            mock_base.close,
            mock_base.volume,
            length,
            num_bins,
            required_length=length
        )

    def test_rolling_vwap(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import rolling_vwap_numba
        length = 14
        volume_indicators.rolling_vwap(length)
        mock_base.calculate_indicator.assert_called_once_with(
            rolling_vwap_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            mock_base.volume,
            length,
            required_length=length
        )

    def test_twap(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import twap_numba
        length = 14
        volume_indicators.twap(length)
        mock_base.calculate_indicator.assert_called_once_with(
            twap_numba,
            mock_base.high,
            mock_base.low,
            mock_base.close,
            length,
            required_length=length
        )

    def test_average_quote_volume(self, mock_base, volume_indicators):
        from src.indicators.volume.volume_indicators import average_quote_volume_numba
        window_size = 14
        volume_indicators.average_quote_volume(window_size)
        mock_base.calculate_indicator.assert_called_once_with(
            average_quote_volume_numba,
            mock_base.close,
            mock_base.volume,
            window_size,
            required_length=window_size
        )
