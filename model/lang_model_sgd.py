from keras import backend as K
from keras.optimizers import Optimizer
from keras.callbacks import LearningRateScheduler
from model.setting import Setting


class LangModelSGD(Optimizer):

    def __init__(self, setting, verbose=True):
        super(LangModelSGD, self).__init__(clipnorm=setting.norm_clipping)
        
        self.iterations = K.variable(0., name="iterations")
        self.lr = K.variable(1.0, name="lr")
        self.epoch_interval = setting.epoch_interval
        self.decay = setting.decay
        self.verbose = verbose

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []
        self.updates.append(K.update_add(self.iterations, 1))
        for p, g in zip(params, grads):
            self.updates.append((p, p - self.lr * g))
        return self.updates

    def get_config(self):
        config = {"iterations": float(K.get_value(self.iterations)),
                  "lr": float(K.get_value(self.lr)),
                  "epoch_interval": int(self.epoch_interval),
                  "decay": float(self.decay)}
        base_config = super(LangModelSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_lr_scheduler(self):
        def scheduler(epoch):
            return 1.0 * self.decay ** (epoch // self.epoch_interval)

        return LearningRateScheduler(scheduler)
