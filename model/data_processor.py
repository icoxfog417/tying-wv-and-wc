import chazutsu
import numpy as np


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
    
    def make_batch_iter(self, r_idx, kind="train", batch_size=20, sentence_size=35, skip=1):
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
        sentence_count = (word_count - sentence_size) / skip + 1
        steps_per_epoch = sentence_count // batch_size

        def generator():
            while True:
                batch_seq_length = (batch_size - 1) * skip + sentence_size
                with open(path, encoding="utf-8") as f:
                    buffer = []
                    sentences = None
                    one_hots = None
                    for line in f:
                        words = r_idx.str_to_ids(line.strip())
                        buffer += words
                        if len(buffer) >= batch_seq_length:
                            _sentences, _one_hots = self.format(buffer, vocab_size, sentence_size, skip=skip)
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


    def format(self, word_seq, vocab_size, sentence_size=35, skip=1):
        sentences = []
        next_words = []
        index = 0
        for i in range(0, len(word_seq) - sentence_size, skip):
            sentences.append(word_seq[i:i + sentence_size])
            nw = word_seq[i + sentence_size]
            next_words.append(nw)
        
        sentences = np.array(sentences)
        one_hots = np.zeros((len(next_words), vocab_size))
        for i, nw in enumerate(next_words):
            one_hots[i][nw] = 1

        return sentences, one_hots
