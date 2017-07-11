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
        dp = DataProcessor()
        samples = np.array([-1] + list(range(10)))
        x, y = dp.format(samples, 10, 5)

        # x          | y
        # ----------------------
        # -1 0 1 2 3 | 0 1 2 3 4
        #  4 5 6 7 8 | 5 6 7 8 9

        self.assertEqual(x.shape, (2, 5))
        self.assertEqual(y.shape, (2, 5, 10))
        for i in range(x.shape[0]):
            self.assertEqual(x[i][1:].tolist(), np.argmax(y[i][:-1], axis=1).flatten().tolist())

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
        max_count = words_in_batch * check_count
        words = []
        with open(r_idx.valid_file_path, encoding="utf-8") as f:
            for line in f:
                words += r_idx.str_to_ids(line.strip())
                if len(words) > max_count:
                    break
        
        for i in range(check_count):
            X, y = next(generator)
            self.assertEqual(X.shape, (batch_size, sequence_size))
            self.assertEqual(y.shape, (batch_size, sequence_size, vocab_size))
            for r in range(X.shape[0]):
                index = i * words_in_batch
                seq = words[index + r*sequence_size:][:sequence_size]
                next_seq = words[index + r*sequence_size + 1:][:sequence_size]
                self.assertEqual(X[r].tolist(), seq)
                self.assertEqual(np.argmax(y[r], axis=1).flatten().tolist(), next_seq)
        
        generator = None
        shutil.rmtree(data_root)


if __name__ == "__main__":
    unittest.main()
