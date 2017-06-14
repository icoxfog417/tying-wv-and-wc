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
        sequence_size = 20
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints")

        dp = DataProcessor()
        test_seq = np.random.randint(vocab_size, size=sequence_size)
        samples = np.tile(test_seq, 10)
        x, y = dp.format(samples, vocab_size, sequence_size)
        x_t, y_t = dp.format(samples, vocab_size, sequence_size, skip=1)

        model = OneHotModel(vocab_size, sequence_size, checkpoint_path=checkpoint_path)
        model.compile()

        model.fit(x, y, x_t, y_t, epochs=20)
        print(model.model.optimizer.get_config())
        pred_seq = np.random.choice(test_seq, 3)
        pred = model.predict(pred_seq)
        # pred will emulates test_seq
        print(test_seq)
        for s, p in zip(pred_seq, pred):
            print("{} -> {}".format(s, p))

        shutil.rmtree(checkpoint_path)

    def test_save_load(self):
        model = OneHotModel(10, 20)
        path = model.save(os.path.dirname(__file__))
        self.assertTrue(os.path.exists(path))
        model.load(path)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
