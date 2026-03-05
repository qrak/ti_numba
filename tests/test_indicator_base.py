import pytest
from src.base.indicator_base import IndicatorBase

def test_get_data_invalid_type():
    indicator = IndicatorBase()

    with pytest.raises(TypeError, match="Data must be a Pandas DataFrame, NumPy array, or List"):
        indicator.get_data("invalid data string")

    with pytest.raises(TypeError, match="Data must be a Pandas DataFrame, NumPy array, or List"):
        indicator.get_data(12345)

    with pytest.raises(TypeError, match="Data must be a Pandas DataFrame, NumPy array, or List"):
        indicator.get_data({"open": 10, "close": 20})
