import pytest
import numpy as np
from src.base.indicator_base import IndicatorBase

import pandas as pd

class TestIndicatorBaseHandleDataFrame:
    def test_handle_dataframe_missing_columns(self):
        """Test that passing a DataFrame with missing required columns raises a ValueError."""
        indicator = IndicatorBase()

        # Create DataFrame missing 'volume'
        data = {
            'open': [10.0, 11.0],
            'high': [12.0, 13.0],
            'low': [9.0, 10.0],
            'close': [11.0, 12.0]
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError, match="Missing columns in DataFrame: {'volume'}"):
            indicator.get_data(df)


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
