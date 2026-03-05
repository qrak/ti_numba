import pytest
import numpy as np
from src.base.indicator_base import IndicatorBase

class TestIndicatorBaseHandleList:
    def test_handle_list_invalid_column_count_less(self):
        """Test that passing a list of lists with fewer elements than expected raises a ValueError."""
        indicator = IndicatorBase()

        # Test with 4 elements (less than expected 5 or 6)
        invalid_data_4 = [[10.0, 11.0, 12.0, 13.0]]

        with pytest.raises(ValueError, match="Each list must contain 5 or 6 elements"):
            indicator.get_data(invalid_data_4)

    def test_handle_list_invalid_column_count_more(self):
        """Test that passing a list of lists with more elements than expected raises a ValueError."""
        indicator = IndicatorBase()

        # Test with 7 elements (more than expected 5 or 6)
        invalid_data_7 = [[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]

        with pytest.raises(ValueError, match="Each list must contain 5 or 6 elements"):
            indicator.get_data(invalid_data_7)

    def test_handle_list_valid_column_count_without_timestamp(self):
        """Test passing a list of lists with 5 elements (open, high, low, close, volume)."""
        indicator = IndicatorBase()

        valid_data_5 = [
            [10.0, 12.0, 9.0, 11.0, 1000.0],
            [11.0, 13.0, 10.0, 12.0, 1500.0]
        ]
        indicator.get_data(valid_data_5)

        assert len(indicator.open) == 2
        assert len(indicator.high) == 2
        assert len(indicator.low) == 2
        assert len(indicator.close) == 2
        assert len(indicator.volume) == 2
        assert indicator.timestamp is None

        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))
        np.testing.assert_array_equal(indicator.close, np.array([11.0, 12.0]))

    def test_handle_list_valid_column_count_with_timestamp(self):
        """Test passing a list of lists with 6 elements (timestamp, open, high, low, close, volume)."""
        indicator = IndicatorBase()

        valid_data_6 = [
            [1672531200.0, 10.0, 12.0, 9.0, 11.0, 1000.0],
            [1672617600.0, 11.0, 13.0, 10.0, 12.0, 1500.0]
        ]
        indicator.get_data(valid_data_6)

        assert indicator.timestamp is not None
        assert len(indicator.timestamp) == 2
        assert len(indicator.open) == 2

        np.testing.assert_array_equal(indicator.timestamp, np.array([1672531200.0, 1672617600.0]))
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))

import pandas as pd
from unittest.mock import patch, MagicMock

class TestIndicatorBaseSaveToCsv:
    @pytest.fixture
    def indicator(self):
        ind = IndicatorBase(save_to_csv=True)
        # 3 data points
        data = [
            [1.0, 2.0, 0.5, 1.5, 100],
            [1.5, 2.5, 1.0, 2.0, 150],
            [2.0, 3.0, 1.5, 2.5, 200]
        ]
        ind.get_data(data)
        return ind

    @pytest.fixture
    def indicator_with_time(self):
        ind = IndicatorBase(save_to_csv=True)
        # 3 data points
        data = [
            [1672531200.0, 1.0, 2.0, 0.5, 1.5, 100],
            [1672617600.0, 1.5, 2.5, 1.0, 2.0, 150],
            [1672704000.0, 2.0, 3.0, 1.5, 2.5, 200]
        ]
        ind.get_data(data)
        return ind

    @patch('src.base.indicator_base.pd.DataFrame')
    def test_save_1d_array_full_length(self, mock_dataframe, indicator):
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance

        result = np.array([10.0, 20.0, 30.0])
        indicator._save_indicator_result_to_csv('test_ind', result)

        # Verify the dictionary passed to DataFrame constructor
        constructor_args = mock_dataframe.call_args[0][0]
        assert 'test_ind' in constructor_args
        np.testing.assert_array_equal(constructor_args['test_ind'], result)
        np.testing.assert_array_equal(constructor_args['open'], indicator.open)
        np.testing.assert_array_equal(constructor_args['close'], indicator.close)

        mock_df_instance.to_csv.assert_called_once_with('test_ind_results.csv', index=False)

    @patch('src.base.indicator_base.pd.DataFrame')
    def test_save_1d_array_single_element(self, mock_dataframe, indicator):
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance

        result = np.array([5.0])
        indicator._save_indicator_result_to_csv('test_ind', result)

        constructor_args = mock_dataframe.call_args[0][0]
        assert 'test_ind' in constructor_args
        # Should be stretched to length 3
        np.testing.assert_array_equal(constructor_args['test_ind'], np.array([5.0, 5.0, 5.0]))

    def test_save_1d_array_invalid_length(self, indicator):
        result = np.array([10.0, 20.0]) # length 2 instead of 3
        with pytest.raises(ValueError, match="Indicator result for test_ind has invalid length 2; expected 3"):
            indicator._save_indicator_result_to_csv('test_ind', result)

    @patch('src.base.indicator_base.pd.DataFrame')
    def test_save_2d_array_n_rows(self, mock_dataframe, indicator):
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance

        # 3 rows, 2 columns
        result = np.array([[1, 2], [3, 4], [5, 6]])
        indicator._save_indicator_result_to_csv('test_ind', result)

        constructor_args = mock_dataframe.call_args[0][0]
        assert 'test_ind_0' in constructor_args
        assert 'test_ind_1' in constructor_args
        np.testing.assert_array_equal(constructor_args['test_ind_0'], np.array([1, 3, 5]))
        np.testing.assert_array_equal(constructor_args['test_ind_1'], np.array([2, 4, 6]))

    @patch('src.base.indicator_base.pd.DataFrame')
    def test_save_2d_array_n_cols(self, mock_dataframe, indicator):
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance

        # 2 rows, 3 columns
        result = np.array([[1, 3, 5], [2, 4, 6]])
        indicator._save_indicator_result_to_csv('test_ind', result)

        constructor_args = mock_dataframe.call_args[0][0]
        assert 'test_ind_0' in constructor_args
        assert 'test_ind_1' in constructor_args
        np.testing.assert_array_equal(constructor_args['test_ind_0'], np.array([1, 3, 5]))
        np.testing.assert_array_equal(constructor_args['test_ind_1'], np.array([2, 4, 6]))

    def test_save_2d_array_invalid_shape(self, indicator):
        result = np.array([[1, 2], [3, 4]]) # 2x2 shape
        with pytest.raises(ValueError, match=r"Indicator result for test_ind has invalid shape \(2, 2\); expected \(3, m\) or \(m, 3\)"):
            indicator._save_indicator_result_to_csv('test_ind', result)

    def test_save_3d_array_invalid_ndim(self, indicator):
        result = np.zeros((3, 3, 3))
        with pytest.raises(ValueError, match="Indicator result for test_ind has invalid number of dimensions: 3"):
            indicator._save_indicator_result_to_csv('test_ind', result)

    @patch('src.base.indicator_base.pd.DataFrame')
    def test_save_tuple_result(self, mock_dataframe, indicator):
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance

        result = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        indicator._save_indicator_result_to_csv('test_ind', result)

        constructor_args = mock_dataframe.call_args[0][0]
        assert 'test_ind_0' in constructor_args
        assert 'test_ind_1' in constructor_args
        np.testing.assert_array_equal(constructor_args['test_ind_0'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(constructor_args['test_ind_1'], np.array([4, 5, 6]))

    @patch('src.base.indicator_base.pd.DataFrame')
    def test_save_with_timestamp(self, mock_dataframe, indicator_with_time):
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance

        result = np.array([10.0, 20.0, 30.0])
        indicator_with_time._save_indicator_result_to_csv('test_ind', result)

        constructor_args = mock_dataframe.call_args[0][0]
        assert 'timestamp' in constructor_args
        np.testing.assert_array_equal(constructor_args['timestamp'], indicator_with_time.timestamp)
