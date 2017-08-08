import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Activation, SpatialDropout1D
from keras import losses
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from model.lang_model_sgd import LangModelSGD
from model.setting import Setting


class OneHotModel():
    
    def __init__(self, 
        vocab_size, 
        sequence_size,
        setting=None,
        checkpoint_path="",
        tensor_board=True):

        self.vocab_size = vocab_size
        self.sequence_size = sequence_size
        self.setting = setting if setting else Setting()
        self.checkpoint_path = checkpoint_path
        self.tensor_board = tensor_board

        dropout = self.setting.dropout
        vector_length = self.setting.vector_length

        self.embedding = Embedding(self.vocab_size, vector_length, input_length=sequence_size)
        layer1 = LSTM(vector_length, return_sequences=True, dropout=dropout)
        layer2 = LSTM(vector_length, return_sequences=True, dropout=dropout)
        softmax_dropout = SpatialDropout1D(dropout)
        projection = TimeDistributed(Dense(self.vocab_size))
        self.model = Sequential()
        self.model.add(self.embedding)
        self.model.add(layer1)
        self.model.add(layer2)
        self.model.add(softmax_dropout)
        self.model.add(projection)
        self.model.add(Activation("softmax"))
    
    def compile(self):
        self.model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=LangModelSGD(self.setting),
            metrics=["accuracy", self.perplexity]
            )
    
    @classmethod
    def perplexity(cls, y_true, y_pred):
        cross_entropy = K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        perplexity = K.exp(cross_entropy)
        return perplexity

    def fit(self, x_train, y_train, x_test, y_test, batch_size=20, epochs=20):
        self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self._get_callbacks()
        )

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
        callbacks = [self.model.optimizer.get_lr_scheduler()]
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

    def predict(self, words):
        x = np.zeros((1, self.sequence_size))  # batch size 1 x sequence_size
        for i, w in enumerate(words):
            if i < self.sequence_size:
                x[0][i] = w
            else:
                break
        pred = self.model.predict(x)[0]  # get first batch's prediction
        # pred.shape = sequence_size x vocab_size
        pred = pred[:len(words)]  # extract next words of given words
        pred_words = np.argmax(pred, axis=1)
        return pred_words
    
    def save(self, folder, suffix=""):
        file_name = self.__class__.__name__.lower() + ("" if not suffix else "_" + suffix) + ".h5"
        path = os.path.join(folder, file_name)
        self.model.save_weights(path)
        return path

    def load(self, path):
        self.model.load_weights(path)
