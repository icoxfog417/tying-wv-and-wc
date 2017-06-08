import os
import numpy as np
from model.one_hot_model import OneHotModel
from model.data_processor import DataProcessor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def flatten(data):
    flatted = []
    for a in data.values.flatten():
        flatted += a
    return np.array(flatted)

def run_ptb(network_size="small"):
    # prepare the data
    dataset_kind = "ptb"
    dp = DataProcessor()
    ptb = dp.get_ptb(DATA_ROOT, vocab_size=10000)
    vocab_size = len(ptb.vocab_data())
    sentence_size = 35
    x_train, y_train = dp.format(flatten(ptb.train_data()), vocab_size, sentence_size)
    x_valid, y_valid = dp.format(flatten(ptb.valid_data()), vocab_size, sentence_size)

    # make one hot model
    model = OneHotModel(vocab_size, sentence_size, network_size, dataset_kind)
    model.compile()
    model.fit(x_valid, y_valid, x_valid, y_valid, epochs=1)


if __name__ == "__main__":
    run_ptb()