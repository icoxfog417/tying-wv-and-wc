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
    
    def make_batch_iter(self, r_idx, kind="train", batch_size=20, sequence_size=35):
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
        steps_per_epoch = 0
        for i in range(sequence_size):
            steps_per_epoch += (word_count - i - 1) // sequence_size
        steps_per_epoch = steps_per_epoch // batch_size

        def generator():
            while True:
                buffer = []
                for i in range(sequence_size):
                    initial_slide = False
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            words = r_idx.str_to_ids(line.strip())
                            if not initial_slide:
                                words = words[i:]
                                initial_slide = True
                            buffer += words
                            if len(buffer) > sequence_size * batch_size:
                                cut_size = sequence_size * batch_size
                                _seq = buffer[:cut_size + 1]  # +1 for next word
                                words, nexts = self.format(_seq, vocab_size, sequence_size)
                                buffer = buffer[cut_size:]

                                yield words, nexts

        return steps_per_epoch, generator()

    def format(self, word_seq, vocab_size, sequence_size):
        words = []
        nexts = []
        sequence_count = (len(word_seq) - 1) // sequence_size
        for i in range(sequence_count):
            start = i * sequence_size
            words.append(word_seq[start:start + sequence_size])
            next_seq = word_seq[(start + 1):(start + 1 + sequence_size)]
            next_seq_as_one_hot = to_categorical(next_seq, vocab_size)  # to one hot vector
            nexts.append(next_seq_as_one_hot)
        
        words = np.array(words)
        nexts = np.array(nexts)

        return words, nexts
