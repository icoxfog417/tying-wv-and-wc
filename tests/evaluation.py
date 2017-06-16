import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import shutil
import argparse
from collections import Counter
import numpy as np
from model.data_processor import DataProcessor
from model.one_hot_model import OneHotModel
from model.augmented_model import AugmentedModel


EVAL_ROOT = os.path.join(os.path.dirname(__file__), "test_evaluation")


def main(kind, epoch):
    if not os.path.exists(EVAL_ROOT):
        os.mkdir(EVAL_ROOT)
    
    sequence_size = 20

    #train_seq = sample_generator(vocab_size, 10000)
    #valid_seq = sample_generator(vocab_size, 2000)
    #test_seq = sample_generator(vocab_size, 20)
    #vocab_size = 100

    words, vocab = read_sentences()
    vocab_size = len(vocab)
    valid_size = int(len(words) / 4)
    train_seq = words[:-valid_size]
    valid_seq = words[-valid_size:-20]
    test_seq = words[-20:]
    print("{} train, {} valid ({} vocab)".format(len(train_seq), len(valid_seq), len(vocab)))

    dp = DataProcessor()
    x, y = dp.format(train_seq, vocab_size, sequence_size)
    x_t, y_t = dp.format(valid_seq, vocab_size, sequence_size)

    if kind == 0:
        print("Build OneHot Model")
        model = OneHotModel(vocab_size, sequence_size, checkpoint_path=EVAL_ROOT)
    elif kind == 1:
        print("Build Augmented Model")
        model = AugmentedModel(vocab_size, sequence_size, checkpoint_path=EVAL_ROOT)
    elif kind == 2:
        print("Build Augmented(Tying) Model")
        model = AugmentedModel(vocab_size, sequence_size, tying=True, checkpoint_path=EVAL_ROOT)
    else:
        raise Exception("Model kind is not specified!")

    model.compile()
    model.fit(x, y, x_t, y_t, epochs=epoch)
    model_pred = model.predict(test_seq)

    rev_vocab =  {v:k for k, v in vocab.items()}
    print([rev_vocab[i] for i in test_seq])
    for s, p in zip(test_seq, model_pred):
        print("{} -> {}".format(rev_vocab[s], rev_vocab[p]))


def sample_generator(vocab_size, count, verbose=False):
    samples = np.random.chisquare(3, count)  # emulate long tail distribution
    samples = samples / np.max(samples)  # 0~1 scale
    samples = np.rint(samples * (vocab_size - 1))  # because includes 0
    if verbose:
        c = Counter()
        for i in samples:
            c[i] += 1
        print(c.most_common())
    return samples


def read_sentences():
    path = os.path.join(os.path.dirname(__file__), "nlp_examples.txt")
    words = []
    vocab = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace(".", "").replace(",", "").replace("`", "").replace(";", "").replace("-", "").strip().lower()
            ws = s.split()
            ws = [w.strip() for w in ws if w.strip()]
            for w in ws:
                if w not in vocab:
                    vocab[w] = len(vocab)
                idx = vocab[w]
                words.append(idx)
    
    print("{} words, {} vocabs.".format(len(words), len(vocab)))

    return words, vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Experiment")
    parser.add_argument("--kind", type=int, default=0, help="model kind (0:lstm, 1:augmented, 2:tying)")
    parser.add_argument("--epoch", default=20, help="train epochs")

    args = parser.parse_args()

    main(args.kind, args.epoch)

