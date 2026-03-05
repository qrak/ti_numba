import pytest
from src.base.indicator_base import IndicatorBase

def test_handle_list_empty():
    base = IndicatorBase()
    with pytest.raises(ValueError, match="Input must be a non-empty list of lists"):
        base.get_data([])

def test_handle_list_invalid_element():
    base = IndicatorBase()
    with pytest.raises(ValueError, match="Input must be a non-empty list of lists"):
        base.get_data([1, 2, 3])
