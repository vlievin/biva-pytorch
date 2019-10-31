import pickle


def deepcopy(model):
    # copy.deepcopy does not work here, here is a workaround
    # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/19
    return pickle.loads(pickle.dumps(model))


class EMA():
    def __init__(self, model, decay):
        self.decay = decay
        self.training_model = model
        if decay > 0:
            self.ema_model = deepcopy(model)
        else:
            self.ema_model = model

    @property
    def model(self):
        return self.ema_model

    def update(self):
        if self.decay == 0:
            return

        for (ema_name, ema_param), (name, param) in zip(self.ema_model.named_parameters(),
                                                        self.training_model.named_parameters()):
            assert ema_name == name
            ema_param.data = self.decay * ema_param.data + (1.0 - self.decay) * param.data
