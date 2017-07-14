import os
import argparse
import numpy as np
from model.one_hot_model import OneHotModel
from model.augmented_model import AugmentedModel
from model.data_processor import DataProcessor
from model.setting import ProposedSetting


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
LOG_ROOT = os.path.join(os.path.dirname(__file__), "log")
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "trained_model")


def prepare_dataset(dataset_kind):
    dp = DataProcessor()
    if dataset_kind == "ptb":
        dataset = dp.get_ptb(DATA_ROOT, vocab_size=10000)
    else:
        dataset = dp.get_wiki2(DATA_ROOT, vocab_size=30000)

    return dataset


def train_baseline(network_size, dataset_kind, epochs=40, stride=0):
    # prepare the data
    setting = ProposedSetting(network_size, dataset_kind)
    dataset = prepare_dataset(dataset_kind)
    vocab_size = len(dataset.vocab_data())
    sequence_size = 20

    dp = DataProcessor()
    train_steps, train_generator = dp.make_batch_iter(dataset, sequence_size=sequence_size, stride=stride)
    valid_steps, valid_generator = dp.make_batch_iter(dataset, kind="valid", sequence_size=sequence_size, stride=stride)

    # make one hot model
    model = OneHotModel(vocab_size, sequence_size, setting, LOG_ROOT)
    model.compile()
    model.fit_generator(train_generator, train_steps, valid_generator, valid_steps, epochs=epochs)
    model.save(MODEL_ROOT)


def train_augmented(network_size, dataset_kind, tying=False, epochs=40, stride=0):
    # prepare the data
    setting = ProposedSetting(network_size, dataset_kind)
    dataset = prepare_dataset(dataset_kind)
    vocab_size = len(dataset.vocab_data())
    sequence_size = 20

    dp = DataProcessor()
    train_steps, train_generator = dp.make_batch_iter(dataset, sequence_size=sequence_size, stride=stride)
    valid_steps, valid_generator = dp.make_batch_iter(dataset, kind="valid", sequence_size=sequence_size, stride=stride)

    # make one hot model
    model = AugmentedModel(vocab_size, sequence_size, setting, tying=tying, checkpoint_path=LOG_ROOT)
    model.compile()
    model.fit_generator(train_generator, train_steps, valid_generator, valid_steps, epochs=epochs)
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
    parser.add_argument("--stride", type=int, default=0, help="stride of the sequence")
    args = parser.parse_args()

    n_size = args.nsize
    dataset = args.dataset

    if not os.path.exists(LOG_ROOT):
        os.mkdir(LOG_ROOT)

    if args.aug or args.tying:
        print("Use Augmented Model (tying={})".format(args.tying))
        train_augmented(n_size, dataset, args.tying, args.epochs, args.stride)
    else:
        train_baseline(n_size, dataset, args.epochs, args.stride)
