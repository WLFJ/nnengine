from tensor import Tensor
import numpy as np


class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self, zero=True):
        raise NotImplemented


class SGD(Optimizer):

    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.alpha = lr

    def zero(self):
        for p in self.parameters:
            p.grad *= 0

    def step(self, zero=True):

        for p in self.parameters:

            p.data -= p.grad * self.alpha

            if (zero):
                p.grad *= 0


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, beta=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.alpha = lr
        self.beta = beta
        self.eps = eps

        self.m = list()
        self.v = list()
        self.t = 0

        for p in self.parameters:
            self.m.append(Tensor(np.zeros_like(p.data)))
            self.v.append(Tensor(np.zeros_like(p.data)))

    def zero(self):
        for p in self.parameters:
            p.grad *= 0

    def step(self, zero=True):

        self.t += 1

        for p, m, v in zip(self.parameters, self.m, self.v):

            m.data = self.beta[0] * m.data + (1 - self.beta[0]) * p.grad
            v.data = self.beta[1] * v.data + (1 - self.beta[1]) * (p.grad ** 2)

            m_hat = m.data / (1 - self.beta[0] ** self.t)
            v_hat = v.data / (1 - self.beta[1] ** self.t)

            p.data -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)

            if (zero):
                p.grad *= 0
