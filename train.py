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


def train(network_size, dataset_kind, kind, layer, batch_size, sequence_size, epochs):
    # prepare the data
    setting = ProposedSetting(network_size, dataset_kind)
    dataset = prepare_dataset(dataset_kind)
    vocab_size = len(dataset.vocab_data())

    # make one hot model
    model = None
    if kind == "onehot":
        model = OneHotModel(vocab_size, sequence_size, layer=layer, batch_size=batch_size, setting=setting, checkpoint_path=LOG_ROOT)
    elif kind == "aug":
        model = AugmentedModel(vocab_size, sequence_size, layer=layer, batch_size=batch_size, setting=setting, checkpoint_path=LOG_ROOT)
    elif kind == "tying":
        model = AugmentedModel(vocab_size, sequence_size, layer=layer, batch_size=batch_size, setting=setting, tying=True, checkpoint_path=LOG_ROOT)
    else:
        raise Exception("Unsupported kind {}".format(kind))

    model.compile()

    dp = DataProcessor()
    train_steps, train_generator = dp.make_batch_iter(
        dataset, batch_size=batch_size, sequence_size=sequence_size
    )
    valid_steps, valid_generator = dp.make_batch_iter(
        dataset, kind="valid", batch_size=batch_size, sequence_size=sequence_size, sequence_end_callback=lambda: model.model.reset_states()
    )

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
    parser.add_argument("--layer", type=int, default=2, help="number of lstm layer")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--seq_size", type=int, default=35, help="sequence size")
    parser.add_argument("--epochs", type=int, default=20, help="epoch to train")
    args = parser.parse_args()

    n_size = args.nsize
    dataset = args.dataset
    kind = "onehot"
    if args.aug:
        kind = "aug"
    if args.tying:
        kind = "tying"

    if not os.path.exists(LOG_ROOT):
        os.mkdir(LOG_ROOT)

    print("Train {} Model".format(kind))
    train(n_size, dataset, kind, args.layer, args.batch_size, args.seq_size, args.epochs)
