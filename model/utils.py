import numpy as np
from keras import backend as K
from keras.engine import InputSpec
from keras.engine import Layer
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.callbacks import Callback


class Bias(Layer):
    def __init__(self, bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Bias, self).__init__(**kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.bias = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def log_perplexity(y_true, y_pred):
    cross_entropy = K.mean(K.categorical_crossentropy(y_true, y_pred), axis=-1)
    return cross_entropy


def perplexity(y_true, y_pred):
    # will be calculated by perplexity logger
    return K.mean(K.zeros_like(y_pred), axis=-1)


class PerplexityLogger(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['perplexity'] = np.exp(logs['log_perplexity'])

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['perplexity'] = np.exp(logs['log_perplexity'])
            logs['val_perplexity'] = np.exp(logs['val_log_perplexity'])
