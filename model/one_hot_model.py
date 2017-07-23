import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Activation, Dropout
from keras import losses
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from model.lang_model_sgd import LangModelSGD
from model.setting import Setting


class OneHotModel():
    
    def __init__(self, 
        vocab_size,
        sequence_size,
        layer=2,
        batch_size=32,
        setting=None,
        checkpoint_path="",
        tensor_board=True):

        self.vocab_size = vocab_size
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.setting = setting if setting else Setting()
        self.checkpoint_path = checkpoint_path
        self.tensor_board = tensor_board

        dropout = self.setting.dropout
        vector_length = self.setting.vector_length

        self.model = Sequential()
        self.embedding = Embedding(self.vocab_size, vector_length, input_length=batch_size, batch_size=1)
        self.model.add(self.embedding)
        for i in range(layer):
            layer = LSTM(vector_length, stateful=True, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
            self.model.add(layer)
        self.model.add(TimeDistributed(Dense(self.vocab_size)))  # projection
        self.model.add(Activation("softmax"))  # to proba
    
    def compile(self):
        self.model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=LangModelSGD(self.setting),
            metrics=["accuracy", self.perplexity]
            )
    
    @classmethod
    def perplexity(cls, y_true, y_pred):
        cross_entropy = K.mean(K.categorical_crossentropy(y_pred, y_true), axis=1)
        perplexity = K.exp(cross_entropy)
        return perplexity

    def fit(self, x_train, y_train, x_test, y_test, epochs=20):

        def train_iter():
            while True:
                for x_t, y_t in zip(x_train, y_train):
                    yield x_t.reshape(1, -1), y_t.reshape(1, self.batch_size, self.vocab_size)

        def test_iter():
            i = 0
            while True:
                for x_t, y_t in zip(x_test, y_test):
                    yield x_t.reshape(1, -1), y_t.reshape(1, self.batch_size, self.vocab_size)
                    i += 1
                    if i % self.sequence_size == 0:
                        self.model.reset_states()
                        i = 0

        self.fit_generator(train_iter(), len(x_train), test_iter(), len(x_test), epochs)

    def fit_generator(self, generator, steps_per_epoch, test_generator, test_steps_per_epoch, epochs=20):
        self.model.fit_generator(
            generator,
            steps_per_epoch,
            validation_data=test_generator,
            validation_steps=test_steps_per_epoch,
            epochs=epochs,
            callbacks=self._get_callbacks()
        )
    
    def _get_callbacks(self):
        callbacks = [self.model.optimizer.get_lr_scheduler(), ResetStatesCallback(self.sequence_size)]
        folder_name = self.get_name()
        self_path = os.path.join(self.checkpoint_path, folder_name)
        if self.checkpoint_path:
            if not os.path.exists(self.checkpoint_path):
                print("Make folder to save checkpoint file to {}".format(self.checkpoint_path))
                os.mkdir(self.checkpoint_path)
            if not os.path.exists(self_path):
                os.mkdir(self_path)

            file_name = "_".join(["model_weights", "{epoch:02d}", "{val_acc:.2f}"]) + ".h5"
            save_callback = ModelCheckpoint(os.path.join(self_path, file_name), save_weights_only=True)
            callbacks += [save_callback]

            if self.tensor_board:
                board_path = os.path.join(self.checkpoint_path, "tensor_board")
                self_board_path = os.path.join(board_path, folder_name)
                if not os.path.exists(board_path):
                    print("Make folder to visualize on TensorBoard to {}".format(board_path))
                    os.mkdir(board_path)
                if not os.path.exists(self_board_path):
                    os.mkdir(self_board_path)
                callbacks += [TensorBoard(self_board_path)]
                print("invoke tensorboard at {}".format(board_path))

        return callbacks

    def get_name(self):
        return self.__class__.__name__.lower()

    def predict(self, words, use_proba=True):
        preds = []
        self.model.reset_states()
        for i, w in enumerate(words):
            x = np.zeros((1, self.batch_size))
            x.fill(w)
            pred = self.model.predict(x)[0]
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

    def save(self, folder, suffix=""):
        file_name = self.__class__.__name__.lower() + ("" if not suffix else "_" + suffix) + ".h5"
        path = os.path.join(folder, file_name)
        self.model.save_weights(path)
        return path

    def load(self, path):
        self.model.load_weights(path)


class ResetStatesCallback(Callback):

    def __init__(self, sequence_size):
        self.sequence_size = sequence_size

    def on_batch_end(self, batch, logs={}):
        if batch % self.sequence_size == 0:
            self.model.reset_states()
