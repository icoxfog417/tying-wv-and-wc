import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Activation, Dropout, Lambda
from model.lang_model_sgd import LangModelSGD
from model.one_hot_model import OneHotModel
from model.settings import DatasetSetting
from keras import backend as K
from keras.losses import kullback_leibler_divergence


class AugmentedModel(OneHotModel):
    
    def __init__(self, 
        vocab_size, 
        sentence_size,
        network_size="small",
        dataset_kind="ptb",
        temperature=20,
        tying=False):

        super().__init__(vocab_size, sentence_size, network_size, dataset_kind)
        self.temperature = temperature
        self.tying = tying
        self.gamma = DatasetSetting.get(dataset_kind)["gamma"]
        self.model.pop()  # remove projection
        temperature = Lambda(lambda x: x / self.temperature)

        if tying:
            self.model.add(Dense(self.vector_length))
            self.model.add(Lambda(lambda x: K.dot(x, K.transpose(self.embedding.embeddings))))
            self.model.add(temperature)
            self.model.add(Activation("softmax"))
        else:
            self.model.add(temperature)
            self.model.add(Dense(self.vocab_size, activation="softmax"))

    def augmented_loss(self, y_true, y_pred):
        loss = K.categorical_crossentropy(y_pred, y_true)
        y_indexes = K.argmax(y_true, axis=1)
        y_vectors = self.embedding(y_indexes)  # y_count x vector_size
        dots = K.map_fn(lambda x: K.dot(self.embedding.embeddings, K.reshape(x, (-1, 1))), y_vectors)
        dots = K.squeeze(dots, axis=2)  # unknown operation
        dots = K.softmax(dots / self.temperature)
        aug_loss = kullback_leibler_divergence(dots, y_pred)
        loss += (self.gamma * self.temperature) * aug_loss

        return loss

    def compile(self):
        sgd = LangModelSGD(self.network_size, self.dataset_kind)
        self.model.compile(
            loss=self.augmented_loss,
            optimizer=sgd,
            metrics=["accuracy", self.perplexity]
            )

    def save(self, folder, suffix=""):
        suffix = "tying" if self.tying else ""
        return super().save(folder, suffix)
