import numpy as np
import pytest

from toynn.core import Tensor, Linear
from toynn.utils import SGD


class TestTensor:
    def test_init(self):
        t = Tensor([1, 2, 3])
        assert (t.data == np.array([1, 2, 3])).all()
        assert (t.grad == np.array([0, 0, 0])).all()

    def test_add(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 + t2
        assert (t3.data == np.array([5, 7, 9])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        t3.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()
        assert (t2.grad == np.array([1, 1, 1])).all()

    def test_sub(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 - t2
        assert (t3.data == np.array([-3, -3, -3])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        t3.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()
        assert (t2.grad == np.array([-1, -1, -1])).all()


class TestNN:
    def test_linear(self):
        data = Tensor(
            np.random.randn(10, 3),
            autograd=False
        )
        m = Linear(3, 1)
        opt = SGD(parameters=m.get_parameters(), lr=0.1)
        pred = m(data)
        for i in range(100):
            loss = ((pred - Tensor(1.)) ** 2.).sum(0)
            loss.backward()
            opt.step()
            pred = m(data)

        assert (pred.data - np.ones((10, 1)) < 1e-5).all()

if __name__ == '__main__':
    pytest.main()
