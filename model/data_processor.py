import chazutsu
import numpy as np
from keras.utils import to_categorical


class DataProcessor():

    def __init__(self):
        pass

    def get_ptb(self, data_root, vocab_size=10000, force=False):
        r = chazutsu.datasets.PTB().download(directory=data_root)
        r_idx = r.to_indexed().make_vocab(vocab_size=vocab_size, force=force)
        return r_idx
    
    def get_wiki2(self, data_root, vocab_size=30000, force=False):
        r = chazutsu.datasets.WikiText2().download(directory=data_root)
        r_idx = r.to_indexed().make_vocab(vocab_size=vocab_size, force=force)
        return r_idx
    
    def make_batch_iter(self, r_idx, kind="train", batch_size=32, sequence_size=35, sequence_end_callback=None):
        # count all tokens
        word_count = 0
        path = r_idx.train_file_path
        if kind == "valid":
            path = r_idx.valid_file_path
        elif kind == "test":
            path = r_idx.test_file_path

        with open(path, encoding="utf-8") as f:
            for line in f:
                words = r_idx.str_to_ids(line.strip())
                word_count += len(words)
        
        vocab_size = len(r_idx.vocab_data())
        words_in_batch = sequence_size * batch_size
        steps_per_epoch = word_count // batch_size

        def generator():
            buffer = []  # do not reset on the ned of the file
            while True:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        words = r_idx.str_to_ids(line.strip())
                        buffer += words
                        if len(buffer) > words_in_batch + batch_size:
                            _seq = buffer[:words_in_batch + batch_size]  # + for next words
                            words, nexts = self.format(_seq, vocab_size, batch_size)
                            for i in range(len(words)):
                                yield words[i].reshape(1, batch_size), nexts[i].reshape(1, batch_size, vocab_size)
                            if sequence_end_callback is not None:
                                sequence_end_callback()
                            buffer = buffer[-batch_size:]

        return steps_per_epoch, generator()

    def format(self, word_seq, vocab_size, batch_size):
        _word_seq = word_seq[:(batch_size * (len(word_seq) // batch_size))]
        _word_seq = np.array(_word_seq).reshape(batch_size, -1)
        _word_seq = np.transpose(_word_seq)
        words = None
        nexts = None
        # iterate over batch size
        for i in range(len(_word_seq) - 1):
            _word = _word_seq[i, :]
            _next = _word_seq[i + 1, :]
            _next = to_categorical(_next, vocab_size)
            if words is None:
                words = _word
                nexts = [_next]
            else:
                words = np.vstack((words, _word))
                nexts.append(_next)

        nexts = np.array(nexts)

        return words, nexts
