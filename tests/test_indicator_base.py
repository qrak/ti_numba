import pytest
import numpy as np
import pandas as pd
from src.base.indicator_base import IndicatorBase

def dummy_func(*args, **kwargs):
    return "result"

def test_calculate_indicator_no_data():
    indicator = IndicatorBase()
    with pytest.raises(ValueError, match=r"Data not initialized\. Call get_data\(\) first\."):
        indicator.calculate_indicator(dummy_func)

def test_calculate_indicator_insufficient_data():
    indicator = IndicatorBase()
    # Populate with 10 rows
    data = np.random.rand(10, 5) # open, high, low, close, volume
    indicator.get_data(data)

    required_length = 20
    expected_message = f"Insufficient data. Need at least {required_length} data points, but only have 10."
    with pytest.raises(ValueError, match=expected_message):
        indicator.calculate_indicator(dummy_func, required_length=required_length)

def test_calculate_indicator_sufficient_data():
    indicator = IndicatorBase()
    data = np.random.rand(20, 5)
    indicator.get_data(data)

    result = indicator.calculate_indicator(dummy_func, required_length=20)
    assert result == "result"
