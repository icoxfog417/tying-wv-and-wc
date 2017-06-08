import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
import numpy as np
from model.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def test_format(self):
        dp = DataProcessor()
        samples = np.array(range(10))
        x, y = dp.format(samples, 5, 3)
        self.assertEqual(x.shape, (7, 3))
        self.assertEqual(y.shape, (7, 5))


if __name__ == "__main__":
    unittest.main()
