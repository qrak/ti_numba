import pytest
import numpy as np
from src.indicators.volume.volume_indicators import obv_numba

def test_obv_numba_basic():
    close = np.array([10.0, 11.0, 10.0, 12.0, 12.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0, 100.0])
    length = 2

    # obv array should be [nan, nan, nan, nan, nan]
    # obv[length-1] = obv[1] = initial(1) * volume[1] = 200
    # i=2: close[2] (10) < close[1] (11) -> obv[2] = obv[1] - volume[2] = 200 - 150 = 50
    # i=3: close[3] (12) > close[2] (10) -> obv[3] = obv[2] + volume[3] = 50 + 300 = 350
    # i=4: close[4] (12) == close[3] (12) -> obv[4] = obv[3] = 350
    expected = np.array([np.nan, 200.0, 50.0, 350.0, 350.0])

    result = obv_numba(close, volume, length)

    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_obv_numba_custom_initial():
    close = np.array([10.0, 11.0, 10.0, 12.0, 12.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0, 100.0])
    length = 2
    initial = -1

    # obv[1] = -1 * 200 = -200
    # obv[2] = -200 - 150 = -350
    # obv[3] = -350 + 300 = -50
    # obv[4] = -50
    expected = np.array([np.nan, -200.0, -350.0, -50.0, -50.0])

    result = obv_numba(close, volume, length, initial=initial)

    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_obv_numba_length_equals_array_length():
    close = np.array([10.0, 11.0])
    volume = np.array([100.0, 200.0])
    length = 2

    expected = np.array([np.nan, 200.0])

    result = obv_numba(close, volume, length)

    np.testing.assert_allclose(result, expected, equal_nan=True)

def test_obv_numba_all_same_close():
    close = np.array([10.0, 10.0, 10.0, 10.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0])
    length = 1

    # length=1 means start at idx 0: obv[0] = 1 * volume[0] = 100
    # i=1: close[1]==close[0] -> obv[1]=obv[0]=100
    # i=2: close[2]==close[1] -> obv[2]=obv[1]=100
    # i=3: close[3]==close[2] -> obv[3]=obv[2]=100
    expected = np.array([100.0, 100.0, 100.0, 100.0])

    result = obv_numba(close, volume, length)

    np.testing.assert_allclose(result, expected, equal_nan=True)
