from keras import backend as K
from keras.optimizers import Optimizer
from keras.callbacks import LearningRateScheduler
from model.setting import Setting


class LangModelSGD(Optimizer):

    def __init__(self, setting, verbose=True):
        super(LangModelSGD, self).__init__(clipnorm=setting.norm_clipping)
        
        self.iterations = K.variable(0., name="iterations")
        self.lr = K.variable(1.0, name="lr")
        self.epoch_interval = K.variable(setting.epoch_interval)
        self.decay = K.variable(setting.decay)
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
