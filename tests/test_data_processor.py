import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import unittest
import numpy as np
import chazutsu
from model.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def test_format(self):
        vocab_size = 10
        sequence_size = 5
        batch_size = 7
        dp = DataProcessor()
        samples = np.array(list(range(sequence_size + 1)) * batch_size)
        x, y = dp.format(samples, vocab_size, sequence_size)

        self.assertEqual(x.shape, (sequence_size, batch_size))
        self.assertEqual(y.shape, (sequence_size, batch_size, vocab_size))
        for i in range(sequence_size):
            self.assertEqual(np.sum(x[i]) / batch_size, i)
            self.assertEqual(np.sum(np.argmax(y[i], axis=-1)) / batch_size, i + 1)

    def test_generator(self):
        data_root = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_root):
            os.mkdir(data_root)

        r = chazutsu.datasets.PTB().download(data_root)
        r_idx = r.to_indexed().make_vocab(vocab_size=10000)

        dp = DataProcessor()
        batch_size = 10
        sequence_size = 15
        vocab_size = len(r_idx.vocab_data())
        steps_per_epoch, generator = dp.make_batch_iter(r_idx, "valid", batch_size, sequence_size)

        words_in_batch = sequence_size * batch_size
        check_count = 5
        max_count = (words_in_batch + sequence_size) * check_count
        words = []
        with open(r_idx.valid_file_path, encoding="utf-8") as f:
            for line in f:
                words += r_idx.str_to_ids(line.strip())
                if len(words) > max_count:
                    break

        for i in range(check_count):
            X, y = next(generator)
            self.assertEqual(X.shape, (sequence_size, batch_size))
            self.assertEqual(y.shape, (sequence_size, batch_size, vocab_size))
            index = i * words_in_batch
            seq = words[index:][:(words_in_batch + batch_size)]
            seq = np.array(seq).reshape((-1, sequence_size + 1))
            seq = np.transpose(seq)

            for r in range(X.shape[0]):
                self.assertEqual(X[r].tolist(), seq[r].tolist())
                self.assertEqual(np.argmax(y[r], axis=-1).tolist(), seq[r + 1].tolist())
        
        generator = None
        shutil.rmtree(data_root)


if __name__ == "__main__":
    unittest.main()
