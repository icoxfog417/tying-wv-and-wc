import copy
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Optimizer
from keras.callbacks import LearningRateScheduler
from model.setting import Setting


class LangModelSGD(Optimizer):

    def __init__(self, setting, verbose=True):
        super(LangModelSGD, self).__init__()
        
        self.iterations = K.variable(0., name="iterations")
        self.lr = K.variable(1.0, name="lr")
        self.epoch_interval = K.variable(setting.epoch_interval)
        self.decay = K.variable(setting.decay)
        self._clipnorm = setting.norm_clipping
        self.verbose = verbose

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [clip_norm(g, self._clipnorm, norm) for g in grads]

        self.updates = []
        self.updates.append(K.update_add(self.iterations, 1))
        for p, g in zip(params, grads):
            self.updates.append((p, p - self.lr * g))
        return self.updates

    def get_config(self):
        config = {"iterations": float(K.get_value(self.iterations)),
                  "lr": float(K.get_value(self.lr))
                  }
        base_config = super(LangModelSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_lr_scheduler(self):
        def scheduler(epoch):
            epoch_interval = K.get_value(self.epoch_interval)
            if epoch != 0 and (epoch + 1) % epoch_interval == 0:
                lr = K.get_value(self.lr)
                decay = K.get_value(self.decay)
                K.set_value(self.lr, lr * decay)
                if self.verbose:
                    print(self.get_config())
            return K.get_value(self.lr)
    
        return LearningRateScheduler(scheduler)


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
