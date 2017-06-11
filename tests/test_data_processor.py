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
        samples = np.array(list(range(5)) * 2)
        x, y = dp.format(samples, 5, 3)
        self.assertEqual(x.shape, (7, 3))
        self.assertEqual(y.shape, (7, 5))

    def test_generator(self):
        data_root = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_root):
            os.mkdir(data_root)

        r = chazutsu.datasets.PTB().download(data_root)
        r_idx = r.to_indexed().make_vocab(vocab_size=10000)

        dp = DataProcessor()
        batch_size = 10
        sentence_size = 15
        skip = 2
        vocab_size = len(r_idx.vocab_data())
        steps_per_epoch, generator = dp.make_batch_iter(r_idx, "valid", batch_size, sentence_size, skip=skip)

        words_in_batch = sentence_size * batch_size
        check_count = 5
        max_count = words_in_batch * check_count
        words = []
        with open(r_idx.valid_file_path, encoding="utf-8") as f:
            for line in f:
                words += r_idx.str_to_ids(line.strip())
                if len(words) > max_count:
                    break
        
        print(words_in_batch)
        index = 0
        for i in range(5):
            X, y = next(generator)
            self.assertEqual(X.shape, (batch_size, sentence_size))
            self.assertEqual(y.shape, (batch_size, vocab_size))
            for r in range(X.shape[0]):
                row = X[r].tolist()
                sen = words[index + (r * skip):][:sentence_size]
                self.assertEqual(row, sen)
            index += batch_size * skip
        
        generator = None
        shutil.rmtree(data_root)


if __name__ == "__main__":
    unittest.main()
