import os
import argparse
import numpy as np
from model.one_hot_model import OneHotModel
from model.augmented_model import AugmentedModel
from model.data_processor import DataProcessor
from model.setting import ProposedSetting


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "trained_model")


def prepare_dataset(dataset_kind):
    dp = DataProcessor()
    if dataset_kind == "ptb":
        dataset = dp.get_ptb(DATA_ROOT, vocab_size=10000)
    else:
        dataset = dp.get_wiki2(DATA_ROOT, vocab_size=30000)

    return dataset


def predict(model_kind, network_size, dataset_kind):
    setting = ProposedSetting(network_size, dataset_kind)
    dataset = prepare_dataset(dataset_kind)
    vocab = dataset.vocab_data()
    vocab_size = len(vocab)
    sequence_size = 20

    test_data = dataset.test_data()

    model = None
    if model_kind == "onehot":
        model = OneHotModel(vocab_size, sequence_size, setting)
    elif model_kind == "aug" or model_kind == "tying":
        model = AugmentedModel(vocab_size, sequence_size, setting, tying=(model_kind == "tying"))
    path = os.path.join(MODEL_ROOT, model.get_name() + ".h5")
    model.load(path)

    test_seq = np.array(test_data.sample(1).iloc[0].values[0])
    model_pred = model.predict(test_seq)
    rev_vocab =  {v:k for k, v in vocab.items()}
    print([rev_vocab[i] for i in test_seq])
    for s, p in zip(test_seq, model_pred):
        print("{} -> {}".format(rev_vocab[s], rev_vocab[p]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the word")
    parser.add_argument("--aug", action="store_const", const=True, default=False,
                        help="use augmented model")
    parser.add_argument("--tying", action="store_const", const=True, default=False,
                        help="use tying model")
    parser.add_argument("--nsize", default="small", help="network size (small, medium, large)")
    parser.add_argument("--dataset", default="ptb", help="dataset kind (ptb or wiki2)")
    args = parser.parse_args()

    n_size = args.nsize
    dataset = args.dataset

    kind = "onehot"
    if args.aug:
        kind = "aug"
    elif args.tying:
        kind = "tying"

    predict(kind, n_size, dataset)
