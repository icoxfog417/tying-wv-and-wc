import os
import argparse
import numpy as np
from model.one_hot_model import OneHotModel
from model.augmented_model import AugmentedModel
from model.data_processor import DataProcessor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "trained_model")


def flatten(data):
    flatted = []
    for a in data.values.flatten():
        flatted += a
    return np.array(flatted)


def train_baseline(network_size, dataset_kind, epochs=40):
    # prepare the data
    dp = DataProcessor()
    ptb = dp.get_ptb(DATA_ROOT, vocab_size=10000, force=True)
    vocab_size = len(ptb.vocab_data())
    sentence_size = 35
    x_train, y_train = dp.format(flatten(ptb.train_data()), vocab_size, sentence_size)
    x_valid, y_valid = dp.format(flatten(ptb.valid_data()), vocab_size, sentence_size)

    # make one hot model
    model = OneHotModel(vocab_size, sentence_size, network_size, dataset_kind)
    model.compile()
    model.fit(x_valid, y_valid, x_valid, y_valid, epochs=epochs)
    model.save(MODEL_ROOT)


def train_augmented(network_size, dataset_kind, tying=False, epochs=40):
    # prepare the data
    dp = DataProcessor()
    ptb = dp.get_ptb(DATA_ROOT, vocab_size=10000)
    vocab_size = len(ptb.vocab_data())
    sentence_size = 35
    x_train, y_train = dp.format(flatten(ptb.train_data()), vocab_size, sentence_size)
    x_valid, y_valid = dp.format(flatten(ptb.valid_data()), vocab_size, sentence_size)

    # make one hot model
    model = AugmentedModel(vocab_size, sentence_size, network_size, dataset_kind, tying=tying)
    model.compile()
    model.fit(x_valid, y_valid, x_valid, y_valid, epochs=epochs)
    model.save(MODEL_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument("--aug", action="store_const", const=True, default=False,
                        help="use augmented model")
    parser.add_argument("--tying", action="store_const", const=True, default=False,
                        help="use tying model")
    parser.add_argument("--nsize", default="small", help="network size (small, medium, large)")
    parser.add_argument("--dataset", default="ptb", help="dataset kind (ptb or wiki2)")
    parser.add_argument("--epochs", type=int, default=40, help="epoch to train")
    args = parser.parse_args()

    n_size = args.nsize
    dataset = args.dataset
    if args.aug or args.tying:
        print("Use Augmented Model (tying={})".format(args.tying))
        train_augmented(n_size, dataset, args.tying, args.epochs)
    else:
        train_baseline(n_size, dataset, args.epochs)
