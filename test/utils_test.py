import pytest
from toynn.core import Tensor, Linear
from toynn.utils import SGD, Adam
import numpy as np


class TestOptimizer:

    def train(self):
        data = Tensor(
            np.random.randn(10, 3),
            autograd=False
        )
        m = Linear(3, 1)
        opt = self.opt(parameters=m.get_parameters(), lr=self.lr)
        pred = m(data)
        for i in range(self.epoch):
            loss = ((pred - Tensor(1.)) ** 2.).sum(0)
            loss.backward()
            opt.step()
            pred = m(data)
        assert (pred.data - np.ones((10, 1)) < 1e-5).all()

    def test_adam(self):
        self.lr = 1e-3
        self.epoch = 7000
        self.opt = Adam
        self.train()

    def test_sgd(self):
        self.lr = 1e-2
        self.epoch = 1000
        self.opt = SGD
        self.train()


if __name__ == '__main__':
    pytest.main()
