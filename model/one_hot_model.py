import os
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Activation, Dropout
from keras import backend as K
from model.lang_model_sgd import LangModelSGD
from model.settings import DatasetSetting


class OneHotModel():
    
    def __init__(self, 
        vocab_size, 
        sentence_size,
        network_size="small",
        dataset_kind="ptb"):

        self.network_size = network_size
        self.dataset_kind = dataset_kind
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.vector_length = self.get_vector_length(network_size)

        dset_setting = DatasetSetting.get(dataset_kind)
        dropout = dset_setting["dropout"][network_size]

        self.embedding = Embedding(self.vocab_size, self.vector_length, input_length=sentence_size)
        #layer1 = LSTM(self.vector_length, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        #layer2 = LSTM(self.vector_length, return_sequences=False, dropout=dropout, recurrent_dropout=dropout)
        cell = LSTM(self.vector_length, dropout=dropout, recurrent_dropout=dropout)
        projection = Dense(self.vocab_size, activation="softmax")
        self.model = Sequential()
        self.model.add(self.embedding)
        self.model.add(cell)
        self.model.add(projection)
    
    def get_vector_length(self, network_size):
        if network_size == "small":
            return 200
        elif network_size == "medium":
            return 650
        elif network_size == "large":
            return 1500
        else:
            return 200

    def compile(self):
        sgd = LangModelSGD(self.network_size, self.dataset_kind)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=sgd,
            metrics=["accuracy", self.perplexity]
            )
    
    @classmethod
    def perplexity(cls, y_true, y_pred):
        cross_entropy = K.categorical_crossentropy(y_pred, y_true)
        perplexity = K.pow(2.0, cross_entropy)
        return perplexity
    
    def fit(self, x_train, y_train, x_test, y_test, batch_size=32, epochs=20):
        self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[self.model.optimizer.get_scheduler()]
        )
    
    def predict(self, words):
        x = np.zeros((1, self.sentence_size))
        for i, w in enumerate(words):
            x[0][i] = w
        pred = self.model.predict(x)[0]
        return pred
    
    def save(self, folder, suffix=""):
        file_name = "_".join([self.__class__.__name__.lower() + suffix]) + ".h5"
        path = os.path.join(folder, file_name)
        self.model.save_weights(path)
        return path

    def load(self, path):
        self.model.load_weights(path)
