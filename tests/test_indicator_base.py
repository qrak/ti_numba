import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.base.indicator_base import IndicatorBase, IndicatorCategory

class TestIndicatorBaseInit:
    def test_initialization(self):
        indicator = IndicatorBase()
        assert indicator.measure_time is False
        assert indicator.save_to_csv is False
        assert indicator.NUM_COLUMNS == 5
        assert indicator.timestamp is None
        np.testing.assert_array_equal(indicator.open, np.array([]))
        np.testing.assert_array_equal(indicator.high, np.array([]))
        np.testing.assert_array_equal(indicator.low, np.array([]))
        np.testing.assert_array_equal(indicator.close, np.array([]))
        np.testing.assert_array_equal(indicator.volume, np.array([]))

class TestIndicatorBaseGetData:
    def test_get_data_invalid_type(self):
        indicator = IndicatorBase()
        with pytest.raises(TypeError, match="Data must be a Pandas DataFrame, NumPy array, or List"):
            indicator.get_data("invalid_data")

class TestIndicatorBaseHandleList:
    def test_handle_list_empty(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="Input must be a non-empty list of lists"):
            indicator.get_data([])

    def test_handle_list_invalid_type(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="Input must be a non-empty list of lists"):
            indicator.get_data([1, 2, 3])

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

class TestIndicatorBaseHandleDataFrame:
    def test_handle_dataframe_missing_columns(self):
        indicator = IndicatorBase()
        df = pd.DataFrame({'Open': [1], 'High': [2], 'Low': [3], 'Close': [4]})
        with pytest.raises(ValueError, match="Missing columns in DataFrame"):
            indicator.get_data(df)

    def test_handle_dataframe_valid(self):
        indicator = IndicatorBase()
        df = pd.DataFrame({
            'Open': [10.0, 11.0],
            'High': [12.0, 13.0],
            'Low': [9.0, 10.0],
            'Close': [11.0, 12.0],
            'Volume': [1000.0, 1500.0]
        })
        indicator.get_data(df)
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))
        np.testing.assert_array_equal(indicator.volume, np.array([1000.0, 1500.0]))
        assert indicator.timestamp is None

    def test_handle_dataframe_with_timestamp(self):
        indicator = IndicatorBase()
        df = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02'],
            'Open': [10.0, 11.0],
            'High': [12.0, 13.0],
            'Low': [9.0, 10.0],
            'Close': [11.0, 12.0],
            'Volume': [1000.0, 1500.0]
        })
        indicator.get_data(df)
        assert indicator.timestamp is not None
        assert len(indicator.timestamp) == 2

class TestIndicatorBaseHandleNumpyArray:
    def test_handle_numpy_invalid_ndim(self):
        indicator = IndicatorBase()
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="NumPy array must be 2-dimensional"):
            indicator.get_data(arr)

    def test_handle_numpy_invalid_cols(self):
        indicator = IndicatorBase()
        arr = np.array([[1, 2, 3, 4]])
        with pytest.raises(ValueError, match="NumPy array must have 5 or 6 columns"):
            indicator.get_data(arr)

    def test_handle_numpy_valid_5_cols(self):
        indicator = IndicatorBase()
        arr = np.array([
            [10.0, 12.0, 9.0, 11.0, 1000.0],
            [11.0, 13.0, 10.0, 12.0, 1500.0]
        ])
        indicator.get_data(arr)
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))
        assert indicator.timestamp is None

    def test_handle_numpy_valid_6_cols(self):
        indicator = IndicatorBase()
        arr = np.array([
            [1672531200.0, 10.0, 12.0, 9.0, 11.0, 1000.0],
            [1672617600.0, 11.0, 13.0, 10.0, 12.0, 1500.0]
        ])
        indicator.get_data(arr)
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))
        assert indicator.timestamp is not None
        np.testing.assert_array_equal(indicator.timestamp, np.array([1672531200.0, 1672617600.0]))

class TestIndicatorBaseCalculateIndicator:
    def test_calculate_not_initialized(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="Data not initialized. Call get_data\\(\\) first."):
            indicator.calculate_indicator(lambda: None)

    def test_calculate_insufficient_data(self):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0]])
        with pytest.raises(ValueError, match="Insufficient data. Need at least 2 data points, but only have 1."):
            indicator.calculate_indicator(lambda: None, required_length=2)

    def test_calculate_measure_time(self, capsys):
        indicator = IndicatorBase(measure_time=True)
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0]])

        def dummy_func(x):
            return x * 2

        res = indicator.calculate_indicator(dummy_func, 5)
        assert res == 10
        captured = capsys.readouterr()
        assert "dummy_func took" in captured.out

    @patch('src.base.indicator_base.IndicatorBase._save_indicator_result_to_csv')
    def test_calculate_save_csv(self, mock_save):
        indicator = IndicatorBase(save_to_csv=True)
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0]])

        def dummy_func():
            return np.array([1.0])

        res = indicator.calculate_indicator(dummy_func)
        np.testing.assert_array_equal(res, np.array([1.0]))
        mock_save.assert_called_once_with('dummy_func', res)

class TestIndicatorBaseSaveToCsv:
    @patch('pandas.DataFrame.to_csv')
    def test_save_1d_array_equal_length(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = np.array([1.0, 2.0])
        indicator._save_indicator_result_to_csv('test_ind', res)
        mock_to_csv.assert_called_once_with('test_ind_results.csv', index=False)

    @patch('pandas.DataFrame.to_csv')
    def test_save_1d_array_single_value(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = np.array([1.0])
        indicator._save_indicator_result_to_csv('test_ind', res)
        mock_to_csv.assert_called_once_with('test_ind_results.csv', index=False)

    def test_save_1d_array_invalid_length(self):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Indicator result for test_ind has invalid length 3; expected 2"):
            indicator._save_indicator_result_to_csv('test_ind', res)

    @patch('pandas.DataFrame.to_csv')
    def test_save_2d_array_rows_match(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = np.array([[1.0, 2.0], [3.0, 4.0]])
        indicator._save_indicator_result_to_csv('test_ind', res)
        mock_to_csv.assert_called_once_with('test_ind_results.csv', index=False)

    @patch('pandas.DataFrame.to_csv')
    def test_save_2d_array_cols_match(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        # Res has length 3 along rows, but 2 along cols
        res = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indicator._save_indicator_result_to_csv('test_ind', res)
        mock_to_csv.assert_called_once_with('test_ind_results.csv', index=False)

    def test_save_2d_array_invalid_shape(self):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        with pytest.raises(ValueError, match="Indicator result for test_ind has invalid shape \\(3, 3\\); expected \\(2, m\\) or \\(m, 2\\)"):
            indicator._save_indicator_result_to_csv('test_ind', res)

    def test_save_invalid_ndim(self):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = np.array([[[1.0]]])
        with pytest.raises(ValueError, match="Indicator result for test_ind has invalid number of dimensions: 3"):
            indicator._save_indicator_result_to_csv('test_ind', res)

    @patch('pandas.DataFrame.to_csv')
    def test_save_tuple_of_arrays(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1500.0]])
        res = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        indicator._save_indicator_result_to_csv('test_ind', res)
        mock_to_csv.assert_called_once_with('test_ind_results.csv', index=False)

    @patch('pandas.DataFrame.to_csv')
    def test_save_with_timestamp(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data([[1672531200.0, 10.0, 12.0, 9.0, 11.0, 1000.0]])
        res = np.array([1.0])
        indicator._save_indicator_result_to_csv('test_ind', res)
        mock_to_csv.assert_called_once_with('test_ind_results.csv', index=False)

class TestIndicatorCategory:
    def test_properties(self):
        base = IndicatorBase()
        base.get_data([[10.0, 12.0, 9.0, 11.0, 1000.0]])
        category = IndicatorCategory(base)

        np.testing.assert_array_equal(category.open, base.open)
        np.testing.assert_array_equal(category.high, base.high)
        np.testing.assert_array_equal(category.low, base.low)
        np.testing.assert_array_equal(category.close, base.close)
        np.testing.assert_array_equal(category.volume, base.volume)
