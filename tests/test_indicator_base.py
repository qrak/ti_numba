import pytest
import numpy as np
from src.base.indicator_base import IndicatorBase

class TestIndicatorBase:
    def test_handle_numpy_array_invalid_dimensions(self):
        indicator_base = IndicatorBase()

        # Pass a 1D array to trigger the exception in _handle_numpy_array
        invalid_data = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="NumPy array must be 2-dimensional"):
            indicator_base.get_data(invalid_data)
