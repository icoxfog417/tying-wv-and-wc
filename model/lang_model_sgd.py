import copy
from keras import backend as K
from keras.optimizers import Optimizer
import numpy as np
import tensorflow as tf
from model.settings import SizeSetting, DatasetSetting


class LangModelSGD(Optimizer):

    def __init__(self, size_kind="small", dataset_kind="ptb"):
        size_setting = SizeSetting.get(size_kind)
        dset_setting = DatasetSetting.get(dataset_kind)
        super(LangModelSGD, self).__init__()
        
        self.iterations = K.variable(0.)
        self.epoch_interval = K.variable(size_setting["epoch_interval"])
        self.lr = K.variable(1.0)
        self.decay = K.variable(size_setting["decay"])
        self._clipnorm = size_setting["norm_clipping"]

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [clip_norm(g, self._clipnorm, norm) for g in grads]
        if self.iterations % self.epoch_interval == 0:
            self.lr = self.lr * self.decay

        self.updates = [(self.iterations, self.iterations + 1.)]
        for p, g in zip(params, grads):
            self.updates.append((p, p - self.lr * g))
        return self.updates

    def get_config(self):
        config = {"lr": float(K.get_value(self.lr)),
                  "decay": float(K.get_value(self.decay)),
                  "epoch_interval": float(K.get_value(self.epoch_interval))
                  }
        base_config = super(LangModelSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_lr(self):
        return self.lr.eval()


# because of https://github.com/fchollet/keras/pull/6859

def clip_norm(g, c, n):
    if c > 0:
        condition = n >= c
        then_expression = tf.scalar_mul(c / n, g)
        else_expression = g

        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, "bool")
        g = tf.cond(condition,
            lambda: then_expression,
            lambda: else_expression)
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape

    return g
