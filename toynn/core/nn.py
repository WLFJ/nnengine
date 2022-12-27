from toynn.core.tensor import Tensor
import numpy as np


class Model(object):

    def __init__(self):
        pass

    def get_parameters(self):
        return list()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


class Sequential(Model):

    def __init__(self, layers=None):
        super().__init__()

        if layers is None:
            layers = []
        self.layers = layers

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
        batch_size, n_inputs, input_height, input_width = input.data.shape
        n_outputs, n_inputs, filter_size, filter_size = self.weight.data.shape

        output_height = int((input_height - filter_size + 2 * self.padding) / self.stride) + 1
        output_width = int((input_width - filter_size + 2 * self.padding) / self.stride) + 1

        self.output = Tensor(np.zeros((batch_size, n_outputs, output_height, output_width)), autograd=True)

        for b in range(batch_size):
            for f in range(n_outputs):
                for i in range(output_height):
                    for j in range(output_width):
                        i_start = i * self.stride
                        i_end = i_start + filter_size
                        j_start = j * self.stride
                        j_end = j_start + filter_size

                        input_region = input.data[b, :, i_start:i_end, j_start:j_end]
                        kernel = self.weight.data[f]
                        if self.bias is not None:
                            self.output.data[b, f, i, j] = (input_region * kernel).sum() + self.bias.data[f]
                        else:
                            self.output.data[b, f, i, j] = (input_region * kernel).sum()
        return self.output


class MaxPool2d(Model):
    def __init__(self, filter_size, stride=None):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride

    def forward(self, input: Tensor):
        batch_size, n_inputs, input_height, input_width = input.data.shape

        output_height = int((input_height - self.filter_size) / self.stride) + 1
        output_width = int((input_width - self.filter_size) / self.stride) + 1

        self.output = Tensor(np.zeros((batch_size, n_inputs, output_height, output_width)), autograd=True)

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    i_start = i * self.stride
                    i_end = i_start + self.filter_size
                    j_start = j * self.stride
                    j_end = j_start + self.filter_size

                    input_region = input.data[b, :, i_start:i_end, j_start:j_end]
                    self.output.data[b, :, i, j] = input_region.max(axes=(1, 2))

        return self.output

    def get_parameters(self):
        return []


class AvgPool2d(Model):
    def __init__(self, filter_size, stride=None):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride

    def forward(self, input: Tensor):
        batch_size, n_inputs, input_height, input_width = input.data.shape

        output_height = int((input_height - self.filter_size) / self.stride) + 1
        output_width = int((input_width - self.filter_size) / self.stride) + 1

        self.output = Tensor(np.zeros((batch_size, n_inputs, output_height, output_width)), autograd=True)

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    i_start = i * self.stride
                    i_end = i_start + self.filter_size
                    j_start = j * self.stride
                    j_end = j_start + self.filter_size

                    input_region = input.data[b, :, i_start:i_end, j_start:j_end]
                    self.output.data[b, :, i, j] = input_region.mean(axis=(1, 2))

        return self.output

    def get_parameters(self):
        return []


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


class Relu(Model):
    def __int__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.relu()

    def get_parameters(self):
        return []
