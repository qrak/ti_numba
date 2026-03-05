import pytest
import pandas as pd
from src.base.indicator_base import IndicatorBase

def test_handle_dataframe_missing_columns():
    indicator = IndicatorBase()

    # Missing 'volume' and 'high'
    df = pd.DataFrame({
        'Open': [10.0, 11.0, 12.0],
        'Low': [9.0, 10.0, 11.0],
        'Close': [11.0, 12.0, 13.0],
    })

    with pytest.raises(ValueError) as excinfo:
        indicator.get_data(df)

    error_msg = str(excinfo.value)
    assert "Missing columns in DataFrame:" in error_msg
    assert "volume" in error_msg
    assert "high" in error_msg

def test_handle_dataframe_missing_one_column():
    indicator = IndicatorBase()

    # Missing 'open'
    df = pd.DataFrame({
        'High': [12.0, 13.0, 14.0],
        'Low': [9.0, 10.0, 11.0],
        'Close': [11.0, 12.0, 13.0],
        'Volume': [1000.0, 1500.0, 2000.0]
    })

    with pytest.raises(ValueError, match=r"Missing columns in DataFrame: \{'open'\}"):
        indicator.get_data(df)
