import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Activation, Dropout, Lambda
from model.lang_model_sgd import LangModelSGD
from model.one_hot_model import OneHotModel
from keras import backend as K
from keras.losses import kullback_leibler_divergence


class AugmentedModel(OneHotModel):
    
    def __init__(self, 
        vocab_size, 
        sequence_size,
        setting=None,
        checkpoint_path="",
        temperature=10,
        tying=False):

        super().__init__(vocab_size, sequence_size, setting, checkpoint_path)
        self.temperature = temperature
        self.tying = tying
        self.gamma = self.setting.gamma

        if tying:
            self.model.pop()  # remove projection
            #self.model.add(TimeDistributed(Dense(self.setting.vector_length)))
            self.model.add(Lambda(lambda x: K.dot(x, K.transpose(self.embedding.embeddings))))
            self.model.add(TimeDistributed(Activation("softmax")))

    def augmented_loss(self, y_true, y_pred):
        loss = K.categorical_crossentropy(y_pred, y_true)
        # y is (batch x seq x vocab)
        y_indexes = K.argmax(y_true, axis=2)  # batch x seq
        y_vectors = self.embedding(y_indexes)  # batch x seq x vector_length
        v_length = self.setting.vector_length
        y_vectors = K.reshape(y_vectors, (-1, v_length))
        y_t = K.map_fn(lambda v: K.dot(self.embedding.embeddings, K.reshape(v, (-1, 1))), y_vectors)
        y_t = K.squeeze(y_t, axis=2)  # unknown but necessary operation
        y_t = K.reshape(y_t, (-1, self.sequence_size, self.vocab_size))
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

    def get_name(self):
        return self.__class__.__name__.lower() + ("_tying" if self.tying else "")

    def save(self, folder, suffix=""):
        suffix = "tying" if self.tying else ""
        return super().save(folder, suffix)
