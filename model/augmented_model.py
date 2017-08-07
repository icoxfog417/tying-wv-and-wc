from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Activation, Dropout, Lambda
from keras import backend as K
from keras.losses import kullback_leibler_divergence
from model.lang_model_sgd import LangModelSGD
from model.one_hot_model import OneHotModel
from model.utils import Bias
from model.utils import perplexity
from model.utils import log_perplexity


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
            self.model.pop()  # remove activation
            self.model.pop()  # remove projection (use self embedding)
            self.model.add(Lambda(lambda x: K.dot(x, K.transpose(self.embedding.embeddings))))
            self.model.add(Bias())
            self.model.add(Activation("softmax"))

    def augmented_loss(self, y_true, y_pred):
        loss = K.categorical_crossentropy(y_true, y_pred)

        # y is (batch x seq x vocab)
        y_indexes = K.argmax(y_true, axis=2)  # turn one hot to index. (batch x seq)
        y_vectors = self.embedding(y_indexes)  # lookup the vector (batch x seq x vector_length)

        # vector x embedding dot products (batch x seq x vocab)
        y_t = K.dot(y_vectors, K.transpose(self.embedding.embeddings))
        y_t = K.softmax(y_t / self.temperature)
        y_pred_t = K.softmax((K.log(y_pred) - self.model.layers[-2].bias) / self.temperature)
        aug_loss = kullback_leibler_divergence(y_t, y_pred_t)
        loss += (self.gamma * self.temperature) * aug_loss
        return loss

    def compile(self):
        self.model.compile(
            loss=self.augmented_loss,
            optimizer=LangModelSGD(self.setting),
            metrics=["accuracy", log_perplexity, perplexity]
            )

    def get_name(self):
        return self.__class__.__name__.lower() + ("_tying" if self.tying else "")

    def save(self, folder, suffix=""):
        suffix = "tying" if self.tying else ""
        return super().save(folder, suffix)
