import numpy as np
import pytest

from lightGE.core import Tensor, Linear, conv2d, max_pool2d, avg_pool2d
from lightGE.utils import SGD, Adam


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
        t3.backward()
        assert (t1.grad == np.array([[15, 15, 15]])).all()
        assert (t2.grad == np.array([[6], [6], [6]])).all()

    def test_exp(self):
        t1 = Tensor([1, 2.], autograd=True)
        t2 = t1.exp()
        assert (t2.data == np.array([np.exp(1), np.exp(2)])).all()
        t2.backward()
        assert (t1.grad == np.array([np.exp(1), np.exp(2)])).all()

    def test_log(self):
        t1 = Tensor([1., 2, 3], autograd=True)
        t2 = t1.log()
        assert (t2.data == np.log(np.array([1., 2, 3]))).all()
        t2.backward()
        assert (t1.grad == 1 / np.array([1., 2, 3])).all()

    def test_sin(self):
        t1 = Tensor([np.pi * 0.5, np.pi, 0], autograd=True)
        t2 = t1.sin()
        assert (abs(t2.data - np.array([1, 0, 0])) <= 1e-10).all()
        t2.backward()
        assert (abs(t1.grad - np.cos([np.pi * 0.5, np.pi, 0])) <= 1e-10).all()

    def test_cos(self):
        t1 = Tensor([np.pi * 0.5, np.pi, 0], autograd=True)
        t2 = t1.cos()
        assert (abs(t2.data - np.array([0, -1, 1])) <= 1e-10).all()
        t2.backward()
        assert (abs(t1.grad + np.sin([np.pi * 0.5, np.pi, 0])) <= 1e-10).all()

    def test_sigmoid(self):
        t1 = Tensor([0., 0, 0], autograd=True)
        t2 = t1.sigmoid()
        assert (t2.data == np.array([0.5, 0.5, 0.5])).all()
        assert (t2.grad == np.array([0, 0, 0])).all()
        t2.backward()
        assert (t1.grad == np.array([0.25, 0.25, 0.25])).all()

    def test_softmax(self):
        t1 = Tensor([1, 2, 3], autograd=True)
        t2 = t1.softmax()
        assert (abs(t2.data - np.array([0.09003057, 0.24472847, 0.66524096])) <= 1e-8).all()
        t2.backward()
        assert (abs(t1.grad == np.array([0.009001, 0.023648, 0.067665])) <= 1e-8).all()

    def test_conv2d(self):
        t1 = Tensor(np.array([[[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]]]), autograd=True)
        t2 = Tensor(np.array([[[[1, 1],
                                [1, 1]]]]), autograd=True)
        t3 = conv2d(t1, t2, stride=1, padding=0)
        assert (t3.data == np.array([[[[12, 16],
                                       [24, 28]]]])).all()
        t3.backward()
        assert (t1.grad == np.array([[[[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]]]])).all()
        assert (t2.grad == np.array([[[[12, 16],
                                       [24, 28]]]])).all()

    def test_maxpool2d(self):
        t1 = Tensor(np.array([[[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]]]), autograd=True)
        t2 = max_pool2d(t1, kernel_size=2, stride=1, padding=0)
        assert (t2.data == np.array([[[[5, 6],
                                       [8, 9]]]])).all()
        t2.backward()
        assert (t1.grad == np.array([[[[0, 0, 0],
                                       [0, 1, 1],
                                       [0, 1, 1]]]])).all()

    def test_avgpool2d(self):
        t1 = Tensor(np.array([[[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]]]), autograd=True)
        t2 = avg_pool2d(t1, kernel_size=2, stride=1, padding=0)
        assert (t2.data == np.array([[[[3, 4],
                                       [6, 7]]]])).all()
        t2.backward()
        assert (t1.grad == np.array([[[[0.25, 0.5, 0.25],
                                       [0.5, 1, 0.5],
                                       [0.25, 0.5, 0.25]]]])).all()

    def test_tanh(self):
        t1 = Tensor([0, 0, 0], autograd=True)
        t2 = t1.tanh()
        assert (t2.data == np.array([0, 0, 0])).all()
        assert (t2.grad == np.array([0, 0, 0])).all()
        t2.backward()
        assert (t1.grad == np.array([1, 1, 1])).all()

    def test_relu(self):
        t1 = Tensor([-1, 2, 0, 4], autograd=True)
        t2 = t1.relu()
        assert (t2.data == np.array([0, 2, 0, 4])).all()
        t2.backward()
        assert (t1.grad == np.array([0, 1, 0, 1])).all()

    def test_abs(self):
        t1 = Tensor([-1, 0, 3], autograd=True)
        t2 = t1.abs()
        assert (t2.data == np.array([1, 0, 3])).all()
        t2.backward()
        assert (t1.grad == np.array([-1, 0, 1])).all()

    def test_sum(self):
        t1 = Tensor([[1, 2, 3]], autograd=True)
        t2 = t1.sum(1)
        assert (t2.data == np.array([6])).all()
        t2.backward()
        assert (t1.grad == np.array([[1, 1, 1]])).all()
        t1.grad *= 0
        t3 = t1.sum(axes=(0, 1))
        assert (t3.data == np.array([6])).all()
        t3.backward()
        assert (t1.grad == np.array([[1, 1, 1]])).all()

    def test_max(self):
        t1 = Tensor([[1, 2, 3]], autograd=True)
        t2 = t1.max(1)
        assert (t2.data == np.array([3])).all()
        t2.backward()
        assert (t1.grad == np.array([[0, 0, 1]])).all()
        t1.grad *= 0
        t3 = t1.max(axes=(0, 1))
        assert (t3.data == np.array([3])).all()
        t3.backward()
        assert (t1.grad == np.array([[0, 0, 1]])).all()

    def test_mean(self):
        data = np.random.randn(2, 3, 4)
        t1 = Tensor(data, autograd=True)
        t2 = t1.mean((0, -1))
        assert (t2.data == data.sum((0, -1)) / 8).all()
        t2.backward()
        assert (t1.grad == np.ones_like(data) / 8).all()

    def test_var(self):
        data = np.random.randn(2, 3, 4)
        t1 = Tensor(data, autograd=True)
        t2 = t1.var((0, -1))
        assert (t2.data == ((data - data.sum((0, 2), keepdims=True) / 8) ** 2).sum((0, -1)) / 8).all()
        t2.backward()
        assert (t1.grad == 2 * (data - data.sum((0, 2), keepdims=True) / 8) / 8).all()

    def test_broadcast(self):
        p1 = np.random.randn(2, 3, 4, 1)
        p2 = np.random.randn(1, 1, 5)
        t1 = Tensor(p1, autograd=True)
        t2 = Tensor(p2, autograd=True)
        t3 = t1 + t2
        assert (t3.data == p1 + p2).all()
        t3.backward()
        assert (t1.grad == 5 * np.ones_like(p1)).all()
        assert (t2.grad == 2 * 3 * 4 * np.ones_like(p2)).all()

    def test_squeeze(self):
        t1 = Tensor([[[0, 0]], [[0, 0]]], autograd=True)
        t2 = t1.squeeze(1)
        assert (t2.data == np.array([[0, 0], [0, 0]])).all()
        t2.backward()
        assert (t1.grad == np.ones_like(t1.data)).all()

    def test_unsqueeze(self):
        t1 = Tensor([[0, 0], [0, 0]], autograd=True)
        t2 = t1.unsqueeze(1)
        assert (t2.data == np.array([[[0, 0]], [[0, 0]]])).all()
        t2.backward()
        assert (t1.grad == np.ones_like(t1.data)).all()

    def test_transpose(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], autograd=True)
        t2 = t1.transpose()
        assert (t2.data == np.array([[1, 4], [2, 5], [3, 6]])).all()
        t3 = t2 * t2
        t3.backward()
        assert (t1.grad == t1.data * 2).all()

    def test_reshape(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], autograd=True)
        t2 = t1.reshape((6,))
        assert (t2.data == np.array([1, 2, 3, 4, 5, 6])).all()
        t2.backward()
        assert (t1.grad == np.ones_like(t1.data)).all()


class TestNN:
    def test_linear(self):
        data = Tensor(
            np.random.randn(10, 3),
            autograd=False
        )
        m = Linear(3, 1)
        opt = SGD(parameters=m.parameters(), lr=0.01)
        pred = m(data)
        for i in range(1000):
            loss = ((pred - Tensor(1.)) ** 2.).sum(0)
            loss.backward()
            opt.step()
            pred = m(data)

        assert (pred.data - np.ones((10, 1)) < 1e-5).all()


if __name__ == '__main__':
    pytest.main()
