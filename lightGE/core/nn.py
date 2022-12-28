from lightGE.core.tensor import Tensor, conv2d, max_pool2d, avg_pool2d
import numpy as np


class Model(object):

    def __init__(self):
        self.is_eval = False
        self.sub_models = []
        pass

    def get_parameters(self):
        return list()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def eval(self):
        self.is_eval = True
        for l in self.sub_models:
            l.eval()

    def train(self):
        self.is_eval = False
        for l in self.sub_models:
            l.train()


class Sequential(Model):

    def __init__(self, layers=None):
        super().__init__()

        if layers is None:
            layers = []
        self.layers = layers

        self.sub_models = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input: Tensor):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


class Linear(Model):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

    def get_parameters(self):
        return [self.weight, self.bias]

    def forward(self, input: Tensor):
        return input.mm(self.weight) + self.bias


class Conv2d(Model):
    def __init__(self, n_inputs, n_outputs, filter_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.weight = Tensor(np.random.randn(n_outputs, n_inputs, filter_size, filter_size) * np.sqrt(
            2.0 / (n_inputs * filter_size * filter_size)), autograd=True)

        if bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        else:
            self.bias = None

    def get_parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def forward(self, input: Tensor):
        output = conv2d(input, self.weight, stride=self.stride, padding=self.padding)

        if self.bias is not None:
            output += self.bias.reshape((1, self.bias.shape[0], 1, 1))

        return output


class MaxPool2d(Model):
    def __init__(self, filter_size, stride=None, padding=0):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor):
        return max_pool2d(input, self.filter_size, self.stride, self.padding)


class AvgPool2d(Model):
    def __init__(self, filter_size, stride=None, padding=0):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor):
        return avg_pool2d(input, self.filter_size, self.stride, self.padding)


class LSTM(Model):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.W_ih = Tensor(np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs), autograd=True)
        self.W_hh = Tensor(np.random.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden), autograd=True)
        self.W_hy = Tensor(np.random.randn(n_hidden, n_outputs) * np.sqrt(2.0 / n_hidden), autograd=True)

        self.bias_h = Tensor(np.zeros(n_hidden), autograd=True)
        self.bias_y = Tensor(np.zeros(n_outputs), autograd=True)

    def forward(self, input: Tensor, hidden: Tensor):
        self.hidden = hidden
        self.input = input

        self.ih = input.mm(self.W_ih) + self.bias_h
        self.hh = self.hidden.mm(self.W_hh)
        self.h = self.ih + self.hh
        self.h.tanh()

        self.y = self.h.mm(self.W_hy) + self.bias_y
        self.y.softmax(dim=1)

        return self.y, self.h

    def get_parameters(self):
        return [self.W_ih, self.W_hh, self.W_hy, self.bias_h, self.bias_y]


class Tanh(Model):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.tanh()

    def get_parameters(self):
        return []


class Sigmoid(Model):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.sigmoid()

    def get_parameters(self):
        return []


class ReLu(Model):
    def __int__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.relu()

    def get_parameters(self):
        return []


class BatchNorm1d(Model):
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs

        self.gamma = Tensor(np.ones(n_inputs), autograd=True)
        self.beta = Tensor(np.zeros(n_inputs), autograd=True)
        self.eps = Tensor(1e-8, autograd=False)

    def forward(self, input: Tensor):
        if self.is_eval:
            return input
        else:
            mean_val = input.mean(0)
            var_val = input.var(0).add(self.eps)
            std_val = var_val.sqrt()

            norm = (input - mean_val) / std_val

            return self.gamma * norm + self.beta

    def get_parameters(self):
        return [self.gamma, self.beta]


class BatchNorm2d(Model):
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs

        self.gamma = Tensor(np.ones(n_inputs), autograd=True)
        self.beta = Tensor(np.zeros(n_inputs), autograd=True)
        self.eps = Tensor(1e-8, autograd=False)

    def forward(self, input: Tensor):
        if self.is_eval:
            return input
        else:
            mean_val = input.mean((0, 2, 3))
            var_val = input.var((0, 2, 3)).add(self.eps)
            std_val = var_val.sqrt()

            norm = (input - mean_val) / std_val

            return self.gamma * norm + self.beta

    def get_parameters(self):
        return [self.gamma, self.beta]


class Dropout(Model):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor):
        if self.is_eval:
            return input
        else:
            self.mask = Tensor(np.random.binomial(1, self.p, size=input.data.shape), autograd=True)
            return input * self.mask

    def get_parameters(self):
        return []


class Dropout2d(Model):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor):
        if self.is_eval:
            return input
        else:
            self.mask = Tensor(np.random.binomial(1, self.p, size=(*input.data.shape[:-2], 1, 1)), autograd=True)
            return input * self.mask

    def get_parameters(self):
        return []
