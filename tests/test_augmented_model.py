import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
import numpy as np
from model.data_processor import DataProcessor
from model.augmented_model import AugmentedModel
from keras import backend as K


class TestAugmentedModel(unittest.TestCase):

    def test_augmented_loss(self):
        vocab_size = 5
        sequence_size = 20
        batch_size = 3

        model = AugmentedModel(vocab_size, sequence_size, batch_size=batch_size)
        y_true = np.array([
            [[1,0,0,0,0]],
            [[0,1,0,0,0]], 
            [[0,0,1,0,0]]
        ])
        y_pred = np.array([
            [[0.8,0,0.1,0,0]],
            [[0,0.9,0,0.3,0]], 
            [[0,0,0.5,0.1,0]]
        ])
        y_true = K.variable(y_true)
        y_pred = K.variable(y_pred)
        loss = model.augmented_loss(y_true, y_pred)
        self.assertTrue(loss is not None)

    def test_model(self):
        vocab_size = 10
        sequence_size = 20
        batch_size = 12

        dp = DataProcessor()
        test_seq = np.random.randint(vocab_size, size=sequence_size + 1)
        samples = np.tile(test_seq, batch_size)
        x, y = dp.format(samples, vocab_size, batch_size)
        x_t, y_t = dp.format(samples, vocab_size, batch_size)

        model = AugmentedModel(vocab_size, sequence_size, layer=1, batch_size=batch_size, )
        model.compile()
        print("augmented model -----------")
        model.fit(x, y, x_t, y_t, epochs=20)

    def test_model_tying(self):
        vocab_size = 10
        sequence_size = 20
        batch_size = 12

        dp = DataProcessor()
        test_seq = np.random.randint(vocab_size, size=sequence_size + 1)
        samples = np.tile(test_seq, batch_size)
        x, y = dp.format(samples, vocab_size, batch_size)
        x_t, y_t = dp.format(samples, vocab_size, batch_size)

        model = AugmentedModel(vocab_size, sequence_size, layer=1, batch_size=batch_size, tying=True)
        model.compile()
        print("tying model ---------------")
        model.fit(x, y, x_t, y_t, epochs=20)
    
    def test_save_load(self):
        model = AugmentedModel(10, 20, tying=True)
        path = model.save(os.path.dirname(__file__))
        self.assertTrue(os.path.exists(path))
        model.load(path)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
