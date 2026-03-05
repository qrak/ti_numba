import numpy as np
import pytest
from src.base.technical_indicators import TechnicalIndicators
from tests.test_indicators import generate_mock_data

def test_performance_array_handling():
    """
    Test array handling and edge cases like NaNs.
    """
    data = generate_mock_data(size=100)
    ind = TechnicalIndicators()
    ind.get_data(data)

    # Make sure we don't crash on NaNs
    data_with_nans = data.copy()
    data_with_nans[50, :] = np.nan
    ind_nan = TechnicalIndicators()
    ind_nan.get_data(data_with_nans)

    try:
        res = ind_nan.trend.adx(length=14)
    except Exception as e:
        pytest.fail(f"Indicator trend.adx failed with NaN input: {str(e)}")
