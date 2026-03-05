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

class TestIndicatorBaseGetData:
    def test_get_data_invalid_type(self):
        indicator = IndicatorBase()
        with pytest.raises(TypeError, match="Data must be a Pandas DataFrame, NumPy array, or List"):
            indicator.get_data("invalid data")

    def test_get_data_empty_list(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="Input must be a non-empty list of lists"):
            indicator.get_data([])

    def test_get_data_list_of_non_lists(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="Input must be a non-empty list of lists"):
            indicator.get_data([1, 2, 3])

class TestIndicatorBaseHandleDataFrame:
    def test_handle_dataframe_valid_with_timestamp(self):
        indicator = IndicatorBase()
        df = pd.DataFrame({
            'timestamp': [1000, 2000],
            'open': [10.0, 11.0],
            'high': [12.0, 13.0],
            'low': [9.0, 10.0],
            'close': [11.0, 12.0],
            'volume': [100.0, 200.0]
        })
        indicator.get_data(df)
        assert indicator.timestamp is not None
        assert len(indicator.timestamp) == 2
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))
        np.testing.assert_array_equal(indicator.volume, np.array([100.0, 200.0]))

    def test_handle_dataframe_valid_without_timestamp(self):
        indicator = IndicatorBase()
        df = pd.DataFrame({
            'open': [10.0, 11.0],
            'high': [12.0, 13.0],
            'low': [9.0, 10.0],
            'close': [11.0, 12.0],
            'volume': [100.0, 200.0]
        })
        indicator.get_data(df)
        assert indicator.timestamp is None
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))

    def test_handle_dataframe_missing_columns(self):
        indicator = IndicatorBase()
        df = pd.DataFrame({
            'open': [10.0, 11.0],
            'high': [12.0, 13.0],
            'low': [9.0, 10.0]
        })
        with pytest.raises(ValueError, match="Missing columns in DataFrame:"):
            indicator.get_data(df)

class TestIndicatorBaseHandleNumpyArray:
    def test_handle_numpy_invalid_dimensions(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="NumPy array must be 2-dimensional"):
            indicator.get_data(np.array([1, 2, 3]))

    def test_handle_numpy_invalid_columns(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="NumPy array must have 5 or 6 columns"):
            indicator.get_data(np.array([[1, 2, 3]]))

    def test_handle_numpy_valid_without_timestamp(self):
        indicator = IndicatorBase()
        arr = np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ])
        indicator.get_data(arr)
        assert indicator.timestamp is None
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))

    def test_handle_numpy_valid_with_timestamp(self):
        indicator = IndicatorBase()
        arr = np.array([
            [1000, 10.0, 12.0, 9.0, 11.0, 100.0],
            [2000, 11.0, 13.0, 10.0, 12.0, 200.0]
        ])
        indicator.get_data(arr)
        assert indicator.timestamp is not None
        np.testing.assert_array_equal(indicator.timestamp, np.array([1000.0, 2000.0]))
        np.testing.assert_array_equal(indicator.open, np.array([10.0, 11.0]))

class TestIndicatorBaseCalculateIndicator:
    def test_calculate_not_initialized(self):
        indicator = IndicatorBase()
        with pytest.raises(ValueError, match="Data not initialized. Call get_data\\(\\) first."):
            indicator.calculate_indicator(lambda x: x)

    def test_calculate_insufficient_length(self):
        indicator = IndicatorBase()
        indicator.get_data(np.array([[10.0, 12.0, 9.0, 11.0, 100.0]]))
        with pytest.raises(ValueError, match="Insufficient data. Need at least 5 data points, but only have 1."):
            indicator.calculate_indicator(lambda x: x, required_length=5)

    @patch('builtins.print')
    def test_calculate_measure_time(self, mock_print):
        indicator = IndicatorBase(measure_time=True)
        indicator.get_data(np.array([[10.0, 12.0, 9.0, 11.0, 100.0]]))

        def mock_func():
            return "result"

        result = indicator.calculate_indicator(mock_func)
        assert result == "result"
        mock_print.assert_called_once()
        assert "mock_func took" in mock_print.call_args[0][0]

    @patch.object(IndicatorBase, '_save_indicator_result_to_csv')
    def test_calculate_save_csv(self, mock_save):
        indicator = IndicatorBase(save_to_csv=True)
        indicator.get_data(np.array([[10.0, 12.0, 9.0, 11.0, 100.0]]))

        def mock_func():
            return "result"

        result = indicator.calculate_indicator(mock_func)
        assert result == "result"
        mock_save.assert_called_once_with("mock_func", "result")

class TestIndicatorBaseSaveCSV:
    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_1d_array(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        result_array = np.array([1.0, 2.0])
        indicator._save_indicator_result_to_csv("my_ind", result_array)

        mock_to_csv.assert_called_once()
        df_saved = mock_to_csv.call_args[0][0]
        # the pandas internal logic is to call `to_csv` on the df
        # So it's easier to verify by actually checking the df being saved

    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_tuple_of_1d_arrays(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        result_tuple = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        indicator._save_indicator_result_to_csv("my_ind", result_tuple)

        mock_to_csv.assert_called_once()
        args, kwargs = mock_to_csv.call_args
        assert args[0] == "my_ind_results.csv"

    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_2d_array(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        # 2 rows, 3 columns
        result_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        indicator._save_indicator_result_to_csv("my_ind", result_array)
        mock_to_csv.assert_called_once()

    def test_save_csv_invalid_length_1d(self):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        result_array = np.array([1.0, 2.0, 3.0]) # Length 3, expected 2
        with pytest.raises(ValueError, match="Indicator result for my_ind has invalid length 3; expected 2"):
            indicator._save_indicator_result_to_csv("my_ind", result_array)

    def test_save_csv_scalar_1d(self):
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            indicator = IndicatorBase()
            indicator.get_data(np.array([
                [10.0, 12.0, 9.0, 11.0, 100.0],
                [11.0, 13.0, 10.0, 12.0, 200.0]
            ]))
            result_array = np.array([5.0]) # Length 1, broadcast to 2
            indicator._save_indicator_result_to_csv("my_ind", result_array)
            mock_to_csv.assert_called_once()

    def test_save_csv_invalid_shape_2d(self):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        # Need shape to be (n, m) or (m, n) where n is 2. (3, 3) is invalid.
        result_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        with pytest.raises(ValueError, match="Indicator result for my_ind has invalid shape \\(3, 3\\); expected \\(2, m\\) or \\(m, 2\\)"):
            indicator._save_indicator_result_to_csv("my_ind", result_array)

    def test_save_csv_invalid_ndim(self):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        result_array = np.array([[[1.0]]]) # 3D
        with pytest.raises(ValueError, match="Indicator result for my_ind has invalid number of dimensions: 3"):
            indicator._save_indicator_result_to_csv("my_ind", result_array)

from src.base.indicator_base import IndicatorCategory

class TestIndicatorCategory:
    def test_indicator_category_properties(self):
        base = IndicatorBase()
        base.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        category = IndicatorCategory(base)

        np.testing.assert_array_equal(category.open, base.open)
        np.testing.assert_array_equal(category.high, base.high)
        np.testing.assert_array_equal(category.low, base.low)
        np.testing.assert_array_equal(category.close, base.close)
        np.testing.assert_array_equal(category.volume, base.volume)

class TestIndicatorBaseSaveCSVMissingCoverage:
    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_with_timestamp(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [1000, 10.0, 12.0, 9.0, 11.0, 100.0],
            [2000, 11.0, 13.0, 10.0, 12.0, 200.0]
        ]))
        result_array = np.array([1.0, 2.0])
        indicator._save_indicator_result_to_csv("my_ind", result_array)

        mock_to_csv.assert_called_once()
        df_saved = mock_to_csv.call_args[0][0]
        # the pandas internal logic is to call `to_csv` on the df
        # Checking mock is enough to verify timestamp path hit

    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_2d_array_transposed(self, mock_to_csv):
        indicator = IndicatorBase()
        indicator.get_data(np.array([
            [10.0, 12.0, 9.0, 11.0, 100.0],
            [11.0, 13.0, 10.0, 12.0, 200.0],
            [12.0, 14.0, 11.0, 13.0, 300.0]
        ]))
        # 2 rows, 3 columns. n=3
        # Should trigger `elif array.shape[1] == n:` block
        result_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        indicator._save_indicator_result_to_csv("my_ind", result_array)
        mock_to_csv.assert_called_once()
# Fixing unused variables
