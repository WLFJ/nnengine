import pytest
from toynn.core import Tensor, Linear
from toynn.utils import SGD, Adam, AdaGrad, Trainer, RMSprop
import numpy as np


class TestOptimizer:

    def train(self):
        p = np.random.randn(1000, 2)
        data = Tensor(p)
        label = Tensor(2) * Tensor(p[:, :1]) + Tensor(p[:, 1:]) + Tensor(-0.5)
        m = Linear(2, 1)
        opt = self.opt(parameters=m.get_parameters(), lr=self.lr)
        pred = m(data)
        for i in range(self.epoch):
            loss = ((pred - label) ** 2.).sum(0)
            loss.backward()
            opt.step()
            pred = m(data)
        assert ((pred.data - label.data) < 1e-5).all()

    def test_Adam(self):
        self.lr = 1e-3
        self.epoch = 10000
        self.opt = Adam
        self.train()

    def test_SGD(self):
        # 对于 SGD 学习率要小一点
        self.lr = 1e-4
        self.epoch = 10000
        self.opt = SGD
        self.train()

    def test_SGDMomentum(self):
        self.lr = 1e-4
        self.epoch = 10000
        self.opt = SGD
        self.train()

    def test_AdaGrad(self):
        # 对于 AdaGrad 学习率要大一点
        self.lr = 1e-1
        self.epoch = 10000
        self.opt = AdaGrad
        self.train()

    def test_RMSpror(self):
        self.lr = 1e-1
        self.epoch = 10000
        self.opt = RMSprop
        self.train()


if __name__ == '__main__':
    pytest.main()
