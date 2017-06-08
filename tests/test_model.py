import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
import numpy as np
from model.data_processor import DataProcessor
from model.one_hot_model import OneHotModel


class TestModel(unittest.TestCase):

    def test_one_hot_forward(self):
        vocab_size = 10
        sentence_size = 20

        dp = DataProcessor()
        samples = np.array(np.random.randint(vocab_size, size=100))
        x, y = dp.format(samples, vocab_size, sentence_size)
        samples = np.array(np.random.randint(vocab_size, size=100))
        x_t, y_t = dp.format(samples, vocab_size, sentence_size)

        model = OneHotModel(vocab_size, sentence_size)
        model.compile()
        model.fit(x, y, x_t, y_t, epochs=1)
        pred = model.predict(np.array([0,1,2]))
        print(pred)


if __name__ == "__main__":
    unittest.main()
