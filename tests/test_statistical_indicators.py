import numpy as np
import pandas as pd
import pytest

from src.indicators.statistical.statistical_indicators import (
    stdev_numba,
    variance_numba,
    zscore_numba,
    mad_numba,
    quantile_numba,
    skew_numba,
    kurtosis_numba,
    entropy_numba,
    hurst_numba,
    linreg_numba,
)

@pytest.fixture
def sample_data():
    """Provides a consistent sample data array for testing."""
    np.random.seed(42)
    # 100 random values between 10 and 20
    return np.random.uniform(10, 20, 100)


def test_stdev_numba(sample_data):
    length = 10
    ddof = 1

    # Calculate expected using pandas
    expected = pd.Series(sample_data).rolling(window=length).std(ddof=ddof).values

    # Calculate actual
    actual = stdev_numba(sample_data, length=length, ddof=ddof)

    # Ignore initial NaN values in comparison
    np.testing.assert_allclose(actual[length-1:], expected[length-1:], rtol=1e-5, atol=1e-8)


def test_variance_numba(sample_data):
    length = 10
    ddof = 1

    # Calculate expected using pandas
    expected = pd.Series(sample_data).rolling(window=length).var(ddof=ddof).values

    # Calculate actual
    actual = variance_numba(sample_data, length=length, ddof=ddof)

    np.testing.assert_allclose(actual[length-1:], expected[length-1:], rtol=1e-5, atol=1e-8)


def test_zscore_numba(sample_data):
    length = 10
    std = 1.0

    # Calculate expected z-score
    s = pd.Series(sample_data)
    rolling_mean = s.rolling(window=length).mean()
    rolling_std = s.rolling(window=length).std(ddof=1)

    # Pandas shift gives the zscore for the *current* value, let's see numba implementation:
    # zscore_values[i] = (close[i] - mean) / (std * stdev)
    # The `mean` and `stdev` in numba are from close[i-length+1 : i+1]
    # So expected is just (s - rolling_mean) / rolling_std

    expected = (s - rolling_mean) / (std * rolling_std)

    actual = zscore_numba(sample_data, length=length, std=std)

    np.testing.assert_allclose(actual[length:], expected.values[length:], rtol=1e-5, atol=1e-8)


def test_mad_numba(sample_data):
    length = 10

    # MAD is mean absolute deviation
    expected = []
    for i in range(len(sample_data)):
        if i < length - 1:
            expected.append(np.nan)
        else:
            window = sample_data[i - length + 1 : i + 1]
            mean = np.mean(window)
            mad = np.mean(np.abs(window - mean))
            expected.append(mad)

    expected = np.array(expected)

    actual = mad_numba(sample_data, length=length)

    np.testing.assert_allclose(actual[length-1:], expected[length-1:], rtol=1e-5, atol=1e-8)


def test_quantile_numba(sample_data):
    length = 10
    q = 0.5 # median

    # Pandas quantile
    expected = pd.Series(sample_data).rolling(window=length).quantile(q).values

    actual = quantile_numba(sample_data, length=length, q=q)

    np.testing.assert_allclose(actual[length-1:], expected[length-1:], rtol=1e-5, atol=1e-8)


def test_skew_numba(sample_data):
    length = 10

    expected = []
    for i in range(len(sample_data)):
        if i < length - 1:
            expected.append(np.nan)
        else:
            window = sample_data[i - length + 1 : i + 1]
            n = len(window)
            mean = np.mean(window)
            std_dev = np.std(window) # using biased standard deviation (ddof=0)
            skew_sum = np.sum(((window - mean) / std_dev) ** 3)
            val = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * skew_sum
            expected.append(val)
    expected = np.array(expected)

    actual = skew_numba(sample_data, length=length)

    np.testing.assert_allclose(actual[length-1:], expected[length-1:], rtol=1e-5, atol=1e-8)


def test_kurtosis_numba(sample_data):
    length = 10

    expected = []
    for i in range(len(sample_data)):
        if i < length - 1:
            expected.append(np.nan)
        else:
            window = sample_data[i - length + 1 : i + 1]
            n = len(window)
            mean = np.mean(window)
            std_dev = np.std(window) # using biased standard deviation (ddof=0)
            kurtosis_sum = np.sum(((window - mean) / std_dev) ** 4)
            kurtosis_constant = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
            kurt = kurtosis_constant * kurtosis_sum
            kurt -= 3 * (n - 1) / ((n - 2) * (n - 3))
            expected.append(kurt)
    expected = np.array(expected)

    actual = kurtosis_numba(sample_data, length=length)

    np.testing.assert_allclose(actual[length-1:], expected[length-1:], rtol=1e-5, atol=1e-8)


def test_entropy_numba(sample_data):
    length = 10

    expected = []
    base = 2.0
    for i in range(len(sample_data)):
        if i < length:
            expected.append(np.nan)
        else:
            # entropy_numba actually uses range(length, n) which means it starts computing at index length.
            # So window is close[i - length:i].
            window = sample_data[i - length : i]
            p = window / np.sum(window)
            ent = -np.sum(p * np.log(p) / np.log(base))
            expected.append(ent)

    expected = np.array(expected)

    actual = entropy_numba(sample_data, length=length)

    np.testing.assert_allclose(actual[length:], expected[length:], rtol=1e-5, atol=1e-8)


def test_hurst_numba(sample_data):
    # Just check it returns valid output without error
    # It takes max_lag = 20, so length should be at least 22
    actual = hurst_numba(sample_data, max_lag=20)

    assert len(actual) == len(sample_data)
    assert np.isnan(actual[:22]).all()
    assert not np.isnan(actual[22:]).any()


def test_linreg_numba(sample_data):
    length = 14

    actual_slope = linreg_numba(sample_data, length=length, r=False)
    actual_r = linreg_numba(sample_data, length=length, r=True)

    # Let's verify against manual slope calculation for the first valid index
    x = np.arange(1, length + 1)
    y = sample_data[0:length]

    slope, intercept = np.polyfit(x, y, 1)

    # linreg_numba slope
    np.testing.assert_allclose(actual_slope[length-1], slope, rtol=1e-5, atol=1e-8)

    # R (correlation coefficient)
    correlation_matrix = np.corrcoef(x, y)
    r = correlation_matrix[0, 1]

    np.testing.assert_allclose(actual_r[length-1], r, rtol=1e-5, atol=1e-8)
