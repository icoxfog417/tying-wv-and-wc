import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import numpy as np
from model.data_processor import DataProcessor
from model.one_hot_model import OneHotModel


class TestOneHotModel(unittest.TestCase):

    def test_one_hot_forward(self):
        vocab_size = 10
        sentence_size = 20
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints")

        dp = DataProcessor()
        samples = np.tile(np.array(np.random.randint(vocab_size, size=sentence_size)), 10)
        x, y = dp.format(samples, vocab_size, sentence_size)
        x_t, y_t = dp.format(samples, vocab_size, sentence_size, skip=1)

        model = OneHotModel(vocab_size, sentence_size, checkpoint_path=checkpoint_path)
        model.compile()
        model.fit(x, y, x_t, y_t, epochs=20)
        print(model.model.optimizer.get_config())
        pred = model.predict(np.array([0,1,2]))
        print(pred)

        shutil.rmtree(checkpoint_path)

    def test_save_load(self):
        model = OneHotModel(10, 20)
        path = model.save(os.path.dirname(__file__))
        self.assertTrue(os.path.exists(path))
        model.load(path)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
