import chazutsu
import numpy as np


class DataProcessor():

    def __init__(self):
        pass

    def get_ptb(self, data_root, vocab_size=10000, force=False):
        r = chazutsu.datasets.PTB().download(directory=data_root)
        r_idx = r.to_indexed().make_vocab(vocab_size=vocab_size, force=force)
        return r_idx
    
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
