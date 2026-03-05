import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.base.indicator_base import IndicatorBase

class TestIndicatorBaseCsvSave(unittest.TestCase):
    def setUp(self):
        self.indicator_base = IndicatorBase(save_to_csv=True)
        # 3 rows of dummy data: [open, high, low, close, volume]
        data = np.array([
            [10.0, 12.0, 9.0, 11.0, 1000.0],
            [11.0, 13.0, 10.0, 12.0, 1500.0],
            [12.0, 14.0, 11.0, 13.0, 2000.0]
        ])
        self.indicator_base.get_data(data)

    def test_save_1d_array_result(self):
        def dummy_indicator():
            return np.array([1.5, 2.5, 3.5])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.indicator_base.calculate_indicator(dummy_indicator)

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(args[0], "dummy_indicator_results.csv")
            self.assertEqual(kwargs.get("index"), False)

    def test_save_tuple_result(self):
        def dummy_tuple_indicator():
            return (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.indicator_base.calculate_indicator(dummy_tuple_indicator)

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(args[0], "dummy_tuple_indicator_results.csv")
            self.assertEqual(kwargs.get("index"), False)

    def test_save_with_timestamp(self):
        self.indicator_base = IndicatorBase(save_to_csv=True)
        data = np.array([
            [1600000000.0, 10.0, 12.0, 9.0, 11.0, 1000.0],
            [1600000060.0, 11.0, 13.0, 10.0, 12.0, 1500.0],
            [1600000120.0, 12.0, 14.0, 11.0, 13.0, 2000.0]
        ])
        self.indicator_base.get_data(data)

        def dummy_indicator_ts():
            return np.array([1.5, 2.5, 3.5])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.indicator_base.calculate_indicator(dummy_indicator_ts)

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(args[0], "dummy_indicator_ts_results.csv")
            self.assertEqual(kwargs.get("index"), False)

    def test_save_scalar_result(self):
        def dummy_scalar_indicator():
            return np.array([42.0])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.indicator_base.calculate_indicator(dummy_scalar_indicator)

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(args[0], "dummy_scalar_indicator_results.csv")
            self.assertEqual(kwargs.get("index"), False)

    def test_save_2d_array_result_shape_0_is_n(self):
        def dummy_2d_indicator():
            return np.array([
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0]
            ])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.indicator_base.calculate_indicator(dummy_2d_indicator)

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(args[0], "dummy_2d_indicator_results.csv")
            self.assertEqual(kwargs.get("index"), False)

    def test_save_2d_array_result_shape_1_is_n(self):
        def dummy_2d_indicator():
            return np.array([
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0]
            ])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.indicator_base.calculate_indicator(dummy_2d_indicator)

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(args[0], "dummy_2d_indicator_results.csv")
            self.assertEqual(kwargs.get("index"), False)

    def test_save_invalid_length_1d_array(self):
        def dummy_invalid_length_indicator():
            return np.array([1.0, 2.0])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            with self.assertRaises(ValueError):
                self.indicator_base.calculate_indicator(dummy_invalid_length_indicator)

    def test_save_invalid_shape_2d_array(self):
        def dummy_invalid_shape_indicator():
            return np.array([
                [1.0, 2.0],
                [3.0, 4.0]
            ])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            with self.assertRaises(ValueError):
                self.indicator_base.calculate_indicator(dummy_invalid_shape_indicator)

    def test_save_invalid_ndim_array(self):
        def dummy_invalid_ndim_indicator():
            return np.array([
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]]
            ])

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            with self.assertRaises(ValueError):
                self.indicator_base.calculate_indicator(dummy_invalid_ndim_indicator)

if __name__ == "__main__":
    unittest.main()
