import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Activation, Dropout, Lambda
from keras import backend as K
import tensorflow as tf
from keras.losses import kullback_leibler_divergence
from model.lang_model_sgd import LangModelSGD
from model.one_hot_model import OneHotModel


class AugmentedModel(OneHotModel):
    
    def __init__(self, 
        vocab_size, 
        sequence_size,
        layer=2,
        batch_size=32,
        setting=None,
        temperature=20,
        tying=False,
        checkpoint_path="",
        tensor_board=True
        ):

        super().__init__(vocab_size, sequence_size, layer, batch_size, setting, checkpoint_path, tensor_board)
        self.temperature = temperature
        self.tying = tying
        self.gamma = self.setting.gamma

        if tying:
            self.model.pop()  # remove projection (use self embedding)
            self.model.pop()  # remove activation
            self.model.add(Lambda(lambda x: K.dot(x, K.transpose(self.embedding.embeddings))))
            self.model.add(Activation("softmax"))

    def augmented_loss(self, y_true, y_pred):
        _y_pred = Activation("softmax")(y_pred)
        loss = K.categorical_crossentropy(_y_pred, y_true)

        # y is (batch x 1 x vocab)
        y_indexes = K.argmax(y_true, axis=2)  # turn one hot to index. (to batch x 1)
        y_vectors = self.embedding(y_indexes)  # lookup the vector (to batch x 1 x vector_length)

        # vector x embedding dot products (to batch x 1 x vocab)
        y_t = tf.tensordot(y_vectors, K.transpose(self.embedding.embeddings), 1)
        y_t = K.reshape(y_t, (self.batch_size, 1, self.vocab_size))  # explicitly set shape
        y_t = K.softmax(y_t / self.temperature)
        _y_pred_t = Activation("softmax")(y_pred / self.temperature)
        aug_loss = kullback_leibler_divergence(y_t, _y_pred_t)
        loss += (self.gamma * self.temperature) * aug_loss
        return loss

    @classmethod
    def perplexity(cls, y_true, y_pred):
        _y_pred = Activation("softmax")(y_pred)
        return super(AugmentedModel, cls).perplexity(y_true, _y_pred)

    def compile(self, optimizer=None):
        self.model.pop()  # remove activation (to calculate aug loss)
        _optimizer = optimizer if optimizer else LangModelSGD(self.setting)

        self.model.compile(
            loss=self.augmented_loss,
            optimizer=_optimizer,
            metrics=["accuracy", self.perplexity]
            )

    def predict(self, words, use_proba=True):
        preds = []
        for i, w in enumerate(words):
            x = np.zeros((self.batch_size, 1))
            x.fill(w)
            pred = self.model.predict(x)[0]
            pred = np.exp(pred) / np.sum(np.exp(pred))
            preds.append(pred)

        preds = np.sum(np.array(preds), axis=1)
        pred_words = []
        for p in preds:
            if use_proba:
                _p = p / np.sum(p)
                w = np.random.choice(range(len(_p)), p=_p)
            else:
                w = np.argmax(p)
            pred_words.append(w)

        return pred_words

    def get_name(self):
        return self.__class__.__name__.lower() + ("_tying" if self.tying else "")

    def save(self, folder, suffix=""):
        suffix = "tying" if self.tying else ""
        return super().save(folder, suffix)
