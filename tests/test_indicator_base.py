import pytest
import numpy as np
from src.base.indicator_base import IndicatorBase

def test_calculate_indicator_uninitialized_data():
    indicator = IndicatorBase()

    with pytest.raises(ValueError, match=r"Data not initialized\. Call get_data\(\) first\."):
        indicator.calculate_indicator(lambda: None)

def test_calculate_indicator_insufficient_data():
    indicator = IndicatorBase()
    # Load 5 data points
    indicator.get_data([
        [1.0, 2.0, 0.5, 1.5, 100],
        [1.5, 2.5, 1.0, 2.0, 200],
        [2.0, 3.0, 1.5, 2.5, 300],
        [2.5, 3.5, 2.0, 3.0, 400],
        [3.0, 4.0, 2.5, 3.5, 500]
    ])

    with pytest.raises(ValueError, match=r"Insufficient data\. Need at least 10 data points, but only have 5\."):
        indicator.calculate_indicator(lambda: None, required_length=10)
