import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Activation, Dropout, Lambda
from model.lang_model_sgd import LangModelSGD
from model.one_hot_model import OneHotModel
from keras import backend as K
from keras.losses import kullback_leibler_divergence


class AugmentedModel(OneHotModel):
    
    def __init__(self, 
        vocab_size, 
        sentence_size,
        setting=None,
        checkpoint_path="",
        temperature=10,
        tying=False):

        super().__init__(vocab_size, sentence_size, setting, checkpoint_path)
        self.temperature = temperature
        self.tying = tying
        self.gamma = self.setting.gamma
        self.verbose = verbose
        self.model.pop()  # remove projection

        if tying:
            self.model.add(Dense(self.setting.vector_length))
            self.model.add(Lambda(lambda x: K.dot(x, K.transpose(self.embedding.embeddings))))
            self.model.add(Activation("softmax"))
        else:
            self.model.add(Dense(self.vocab_size, activation="softmax"))

    def augmented_loss(self, y_true, y_pred):
        loss = K.categorical_crossentropy(y_pred, y_true)
        y_indexes = K.argmax(y_true, axis=1)
        y_vectors = self.embedding(y_indexes)  # y_count x vector_size
        y_t = K.map_fn(lambda v: K.dot(self.embedding.embeddings, K.reshape(v, (-1, 1))), y_vectors)
        y_t = K.squeeze(y_t, axis=2)  # unknown operation
        y_t = K.softmax(y_t / self.temperature)
        aug_loss = kullback_leibler_divergence(y_t, y_pred)
        loss += (self.gamma * self.temperature) * aug_loss
        return loss

    def compile(self):
        self.model.compile(
            loss=self.augmented_loss,
            optimizer=LangModelSGD(self.setting),
            metrics=["accuracy", self.perplexity]
            )

    def save(self, folder, suffix=""):
        suffix = "tying" if self.tying else ""
        return super().save(folder, suffix)
