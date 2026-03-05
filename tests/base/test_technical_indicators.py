import unittest
import numpy as np
import pandas as pd

from src.base.technical_indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        self.ti = TechnicalIndicators(measure_time=True, save_to_csv=True)

    def test_initialization(self):
        # Verify base initialization
        self.assertTrue(self.ti._base.measure_time)
        self.assertTrue(self.ti._base.save_to_csv)

        # Verify indicator categories initialization
        self.assertIsNotNone(self.ti.overlap)
        self.assertIsNotNone(self.ti.momentum)
        self.assertIsNotNone(self.ti.price)
        self.assertIsNotNone(self.ti.sentiment)
        self.assertIsNotNone(self.ti.statistical)
        self.assertIsNotNone(self.ti.support_resistance)
        self.assertIsNotNone(self.ti.trend)
        self.assertIsNotNone(self.ti.volatility)
        self.assertIsNotNone(self.ti.vol)

        # Ensure momentum has access to overlap
        self.assertEqual(self.ti.momentum.overlap, self.ti.overlap)

    def test_property_delegation(self):
        # Set dummy data in _base
        self.ti._base.open = np.array([1.0, 2.0])
        self.ti._base.high = np.array([3.0, 4.0])
        self.ti._base.low = np.array([0.5, 1.5])
        self.ti._base.close = np.array([2.0, 3.0])
        self.ti._base.volume = np.array([100.0, 200.0])

        # Test delegation
        np.testing.assert_array_equal(self.ti.open, self.ti._base.open)
        np.testing.assert_array_equal(self.ti.high, self.ti._base.high)
        np.testing.assert_array_equal(self.ti.low, self.ti._base.low)
        np.testing.assert_array_equal(self.ti.close, self.ti._base.close)
        np.testing.assert_array_equal(self.ti.volume, self.ti._base.volume)

    def test_get_data_numpy(self):
        # 5 columns: open, high, low, close, volume
        data = np.array([
            [1.0, 2.0, 0.5, 1.5, 100.0],
            [2.0, 3.0, 1.5, 2.5, 200.0]
        ])

        self.ti.get_data(data)

        np.testing.assert_array_equal(self.ti.open, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(self.ti.high, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(self.ti.low, np.array([0.5, 1.5]))
        np.testing.assert_array_equal(self.ti.close, np.array([1.5, 2.5]))
        np.testing.assert_array_equal(self.ti.volume, np.array([100.0, 200.0]))

    def test_get_data_pandas(self):
        df = pd.DataFrame({
            'open': [1.0, 2.0],
            'high': [2.0, 3.0],
            'low': [0.5, 1.5],
            'close': [1.5, 2.5],
            'volume': [100.0, 200.0]
        })

        self.ti.get_data(df)

        np.testing.assert_array_equal(self.ti.open, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(self.ti.high, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(self.ti.low, np.array([0.5, 1.5]))
        np.testing.assert_array_equal(self.ti.close, np.array([1.5, 2.5]))
        np.testing.assert_array_equal(self.ti.volume, np.array([100.0, 200.0]))

    def test_get_data_list(self):
        data = [
            [1.0, 2.0, 0.5, 1.5, 100.0],
            [2.0, 3.0, 1.5, 2.5, 200.0]
        ]

        self.ti.get_data(data)

        np.testing.assert_array_equal(self.ti.open, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(self.ti.high, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(self.ti.low, np.array([0.5, 1.5]))
        np.testing.assert_array_equal(self.ti.close, np.array([1.5, 2.5]))
        np.testing.assert_array_equal(self.ti.volume, np.array([100.0, 200.0]))

    def test_get_data_empty(self):
        # Empty properties before get_data
        ti_empty = TechnicalIndicators()
        self.assertEqual(len(ti_empty.open), 0)
        self.assertEqual(len(ti_empty.high), 0)
        self.assertEqual(len(ti_empty.low), 0)
        self.assertEqual(len(ti_empty.close), 0)
        self.assertEqual(len(ti_empty.volume), 0)

    def test_get_data_invalid(self):
        # Pass a completely invalid object to get_data and expect TypeError
        with self.assertRaises(TypeError):
            self.ti.get_data("invalid data type")

if __name__ == '__main__':
    unittest.main()
