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
        sentence_size = 20

        model = AugmentedModel(vocab_size, sentence_size)
        y_true = np.array([[1,0,0,0,0],[0,1,0,0,0], [0,0,1,0,0]])
        y_pred = np.array([[0.8,0.1,0,0,0],[0,0.9,0.1,0,0], [0,0.2,0.9,0,0]])
        y_true = K.variable(y_true)
        y_pred = K.variable(y_pred)
        loss = model.augmented_loss(y_true, y_pred)
        self.assertTrue(loss is not None)

    def test_model(self):
        vocab_size = 10
        sentence_size = 20

        dp = DataProcessor()
        samples = np.array(np.random.randint(vocab_size, size=100))
        x, y = dp.format(samples, vocab_size, sentence_size)
        samples = np.array(np.random.randint(vocab_size, size=100))
        x_t, y_t = dp.format(samples, vocab_size, sentence_size)

        model = AugmentedModel(vocab_size, sentence_size)
        model.compile()
        model.fit(x, y, x_t, y_t, epochs=1)

    def test_model_tying(self):
        vocab_size = 10
        sentence_size = 20

        dp = DataProcessor()
        samples = np.array(np.random.randint(vocab_size, size=100))
        x, y = dp.format(samples, vocab_size, sentence_size)
        samples = np.array(np.random.randint(vocab_size, size=100))
        x_t, y_t = dp.format(samples, vocab_size, sentence_size)

        model = AugmentedModel(vocab_size, sentence_size, tying=True)
        model.compile()
        model.fit(x, y, x_t, y_t, epochs=1)
    
    def test_save_load(self):
        model = AugmentedModel(10, 20, tying=True)
        path = model.save(os.path.dirname(__file__))
        self.assertTrue(os.path.exists(path))
        model.load(path)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
