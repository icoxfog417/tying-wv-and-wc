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
    
    def make_batch_iter(self, r_idx, kind="train", batch_size=20, sequence_size=35, skip=1):
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
        sentence_count = (word_count - sequence_size) / skip + 1
        steps_per_epoch = sentence_count // batch_size

        def generator():
            while True:
                batch_seq_length = (batch_size - 1) * skip + sequence_size
                with open(path, encoding="utf-8") as f:
                    buffer = []
                    sentences = None
                    one_hots = None
                    for line in f:
                        words = r_idx.str_to_ids(line.strip())
                        buffer += words
                        if len(buffer) >= batch_seq_length:
                            _sentences, _one_hots = self.format(buffer, vocab_size, sequence_size, skip=skip)
                            buffer = buffer[(len(_sentences) * skip):]

                            if sentences is None:
                                sentences = _sentences
                                one_hots = _one_hots
                            else:
                                sentences = np.vstack((sentences, _sentences))
                                one_hots = np.vstack((one_hots, _one_hots))

                            while len(sentences) >= batch_size:
                                yield sentences[:batch_size, :], one_hots[:batch_size, :]
                                sentences = sentences[batch_size:, :]
                                one_hots = one_hots[batch_size:, :]
        return steps_per_epoch, generator()


    def format(self, word_seq, vocab_size, sequence_size=35, skip=1):
        sentences = []
        next_words = []
        index = 0
        for i in range(0, len(word_seq) - sequence_size, skip):
            sentences.append(word_seq[i:i + sequence_size])
            n_words = word_seq[(i + 1):(i + 1 + sequence_size)]
            n_words = to_categorical(n_words, vocab_size)  # to one hot vector
            next_words.append(n_words)
        
        sentences = np.array(sentences)
        next_words = np.array(next_words)

        return sentences, next_words
