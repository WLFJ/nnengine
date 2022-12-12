import numpy as np
import pytest
import math

from toynn.core import Tensor, Linear
from toynn.utils import SGD, Adam


class TestTensor:
    def test_init(self):
        t = Tensor([1, 2, 3], autograd=True)
        assert (t.data == np.array([1, 2, 3])).all()
        assert (t.grad == np.array([0, 0, 0])).all()

    def test_add(self):
        t1 = Tensor([1, 2, 3], autograd=True)
        t2 = Tensor([4, 5, 6], autograd=True)
        t3 = t1 + t2
        assert (t3.data == np.array([5, 7, 9])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        t3.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()
        assert (t2.grad == np.array([1, 1, 1])).all()

    def test_sub(self):
        t1 = Tensor([1, 2, 3], autograd=True)
        t2 = Tensor([4, 5, 6], autograd=True)
        t3 = t1 - t2
        assert (t3.data == np.array([-3, -3, -3])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        t3.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()
        assert (t2.grad == np.array([-1, -1, -1])).all()

    def test_neg(self):
        t1 = Tensor([1, 2, 3], autograd=True)
        t2 = -t1
        assert (t2.data == np.array([-1, -2, -3])).all()
        assert (t2.grad == np.array([0, 0, 0])).all()
        t2.backward()
        assert (t1.grad == np.array([-1, -1, -1])).all()

    def test_mul(self):
        t1 = Tensor([[1, 2, 3]], autograd=True)
        t2 = Tensor([[4],
                     [5],
                     [6]], autograd=True)
        t3 = t1 * t2
        assert (t3.data == np.array([[4, 8, 12],
                                     [5, 10, 15],
                                     [6, 12, 18]])).all()
        assert (t3.grad == np.array([0])).all()

    def test_exp(self):
        t1 = Tensor([0, math.log(2.)], autograd=True)
        t2 = t1.exp()
        assert (t2.data == np.array([1., 2.])).all()

    def test_log(self):
        t1 = Tensor([math.exp(2), math.exp(4), math.exp(3)], autograd=True)
        t2 = t1.log()
        assert (t2.data == np.array([2, 4, 3])).all()

    def test_sin(self):
        t1 = Tensor([math.pi * 0.5, math.pi, 0], autograd=True)
        t2 = t1.sin()
        print(t2)
        assert (abs(t2.data - np.array([1, 0, 0])) <= np.array([1.0000000e+00, 1.2246468e-16, 0.0000000e+00])).all()

    def test_cos(self):
        t1 = Tensor([math.pi * 0.5, math.pi, 0], autograd=True)
        t2 = t1.cos()
        print(t2)
        assert (abs(t2.data - np.array([0, -1, 1])) <= np.array([1.0000000e+00, 1.2246468e-16, 0.0000000e+00])).all()

    def test_sigmoid(self):
        t1 = Tensor([0, 0, 0], autograd=True)
        t2 = t1.sigmoid()
        assert (t2.data == np.array([0.5, 0.5, 0.5])).all()
        assert (t2.grad == np.array([0, 0, 0])).all()
        # t2.backward()
        # assert (t1.grad == np.array([0.25, 0.25, 0.25])).all()

    def test_tanh(self):
        t1 = Tensor([0, 0, 0], autograd=True)
        t2 = t1.tanh()
        assert (t2.data == np.array([0, 0, 0])).all()
        assert (t2.grad == np.array([0, 0, 0])).all()

    def test_relu(self):
        t1 = Tensor([1, 2, 3], autograd=True)
        t2 = Tensor([-1, -2, -3], autograd=True)
        t3 = t1.relu()
        t4 = t2.relu()
        assert (t3.data == np.array([1, 2, 3])).all()
        assert (t3.grad == np.array([0, 0, 0])).all()
        assert (t4.data == np.array([0, 0, 0])).all()
        assert (t4.grad == np.array([0, 0, 0])).all()

    def test_abs(self):
        t1 = Tensor([-1, -2, -3], autograd=True)
        t2 = t1.abs()
        assert (t2.data == np.array([1, 2, 3])).all()

    def test_sum(self):
        t1 = Tensor([[1, 2, 3]], autograd=True)
        t2 = t1.sum(1)
        assert (t2.data == np.array([6])).all()

    def test_max(self):
        t1 = Tensor([[1, 2, 3],
                     [4, 5, 6]], autograd=True)
        t2 = t1.max(1)
        t3 = t1.max(0)
        print(t2)
        assert (t2.data == np.array([3, 6])).all()
        assert (t3.data == np.array([4, 5, 6])).all()

    # def test_softmax(self):

    def test_broadcast(self):
        t1 = Tensor([1, 2], autograd=True)
        t2 = Tensor([[1], [2], [3]], autograd=True)
        t3 = t1 + t2
        assert (t3.data == np.array([[2, 3], [3, 4], [4, 5]])).all()

    def test_squeeze(self):
        t1 = Tensor([[[0, 0]], [[0, 0]]], autograd=True)
        t2 = t1.squeeze(1)
        assert (t2.data == np.array([[0, 0], [0, 0]])).all()

    def test_unsqueeze(self):
        t1 = Tensor([[0, 0], [0, 0]], autograd=True)
        t2 = t1.unsqueeze(1)
        assert (t2.data == np.array([[[0, 0]], [[0, 0]]])).all()

    def test_transpose(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], autograd=True)
        t2 = t1.transpose()
        assert (t2.data == np.array([[1, 4], [2, 5], [3, 6]])).all()

    def test_reshape(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], autograd=True)
        t2 = t1.reshape(6)
        print(t2)
        assert (t2.data == np.array([1, 2, 3, 4, 5, 6])).all()


class TestNN:
    def test_linear(self):
        data = Tensor(
            np.random.randn(10, 3),
            autograd=False
        )
        m = Linear(3, 1)
        opt = SGD(parameters=m.get_parameters(), lr=0.01)
        pred = m(data)
        for i in range(1000):
            loss = ((pred - Tensor(1.)) ** 2.).sum(0)
            loss.backward()
            opt.step()
            pred = m(data)

        assert (pred.data - np.ones((10, 1)) < 1e-5).all()


if __name__ == '__main__':
    pytest.main()
