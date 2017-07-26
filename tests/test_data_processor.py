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
        x, y = dp.format(samples, vocab_size, batch_size)

        self.assertTrue(len(x) == len(y) == sequence_size)
        self.assertEqual(x[0].shape, (batch_size, 1))
        self.assertEqual(y[0].shape, (batch_size, 1, vocab_size))

        for i in range(sequence_size):
            self.assertEqual(x[i].flatten().tolist(), [i] * batch_size)
            n = np.argmax(np.squeeze(y[i], axis=1), axis=1)
            self.assertEqual([i + 1] * batch_size, n.tolist())

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
        max_count = (words_in_batch + batch_size) * check_count
        words = []
        with open(r_idx.valid_file_path, encoding="utf-8") as f:
            for line in f:
                words += r_idx.str_to_ids(line.strip())
                if len(words) > max_count:
                    words = words[:max_count]

        words = np.array(words)
        subseq = None
        for i in range(len(words) // batch_size - 1):
            if subseq is None or i % sequence_size == 0:
                index = i // sequence_size
                subseq = words[index * words_in_batch:][:words_in_batch + batch_size]
                subseq = subseq.reshape(batch_size, -1)
                if index >= check_count:
                    break

            subseq_index = i % sequence_size
            X, y = next(generator)                

            self.assertEqual(X.shape, (batch_size, 1))
            self.assertEqual(y.shape, (batch_size, 1, vocab_size))

            X_ans = subseq[:, subseq_index]
            y_ans = subseq[:, subseq_index + 1]

            self.assertEqual(X.flatten().tolist(), X_ans.tolist())
            self.assertEqual(np.argmax(np.squeeze(y, axis=1), axis=1).tolist(), y_ans.tolist())

        generator = None
        shutil.rmtree(data_root)


if __name__ == "__main__":
    unittest.main()
