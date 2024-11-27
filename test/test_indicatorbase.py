import unittest

import numpy as np
import pandas as pd

from src.ti_numba.base import IndicatorBase


class TestIndicatorBase(unittest.TestCase):
    def setUp(self) -> None:
        self.indicator = IndicatorBase(measure_time=True, save_to_csv=True)
        self.sample_data = np.array([
            [1.0, 2.0, 1.0, 1.5, 1000.0],
            [2.0, 3.0, 1.5, 2.5, 2000.0],
            [3.0, 4.0, 2.0, 3.5, 3000.0]
        ])
        print(f"\nRunning: {self._testMethodName}")

    def tearDown(self) -> None:
        print(f"Completed: {self._testMethodName}")

    def test_initialization(self) -> None:
        self.assertTrue(self.indicator.measure_time)
        self.assertTrue(self.indicator.save_to_csv)
        self.assertEqual(self.indicator.NUM_COLUMNS, 5)
        self.assertIsNone(self.indicator.timestamp)
        print("Initialization test passed")

    def test_numpy_input(self) -> None:
        self.indicator.get_data(self.sample_data)
        np.testing.assert_array_equal(self.indicator.close.ravel(), self.sample_data[:, 3])
        np.testing.assert_array_equal(self.indicator.volume.ravel(), self.sample_data[:, 4])
        print("Numpy input test passed")

    def test_dataframe_input(self) -> None:
        df = pd.DataFrame(
            self.sample_data,
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        self.indicator.get_data(df)
        np.testing.assert_array_equal(self.indicator.close.ravel(), df['close'].values)
        print("DataFrame input test passed")

    def test_list_input(self) -> None:
        data_list = self.sample_data.tolist()
        self.indicator.get_data(data_list)
        np.testing.assert_array_equal(self.indicator.close.ravel(), np.array([1.5, 2.5, 3.5]))
        print("List input test passed")

    def test_invalid_input(self) -> None:
        with self.assertRaises(TypeError):
            self.indicator.get_data("invalid")
        print("Invalid input test passed")

    def test_calculate_indicator(self) -> None:
        self.indicator.get_data(self.sample_data)

        def dummy_func(x: np.ndarray) -> np.ndarray:
            return x * 2

        result = self.indicator.calculate_indicator(
            dummy_func,
            self.indicator.close,
            required_length=2
        )
        np.testing.assert_array_equal(result.ravel(), self.indicator.close.ravel() * 2)
        print("Calculate indicator test passed")

    def test_insufficient_data(self) -> None:
        self.indicator.get_data(self.sample_data)

        with self.assertRaises(ValueError):
            self.indicator.calculate_indicator(
                lambda x: x,
                self.indicator.close,
                required_length=10
            )
        print("Insufficient data test passed")

    def test_timestamp_handling(self) -> None:
        df = pd.DataFrame(
            self.sample_data,
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df))
        self.indicator.get_data(df)
        self.assertIsNotNone(self.indicator.timestamp)
        print("Timestamp handling test passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
