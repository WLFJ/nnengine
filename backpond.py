import numpy as np

from toynn.core.tensor import Tensor
from toynn.utils import SGD, Adam


def add_backprop_test():
    t1 = Tensor([[1, 2, 3, 4],
                 [1, 2, 3, 4.]])

    t2 = Tensor([[1, 2, 3, 4],
                 [1, 2, 3, 4.]])

    t3 = t1 + t2
    opt = SGD([t1, t2], lr=0.1)

    t4 = Tensor([[3., 6, 9, 12],
                 [3., 6, 9, 12.]])

    (t3 - t4).abs().backward()

    opt.step()

    print(t2)
    print(t1)


def sum_backprop_test():
    t1 = Tensor([1., 2, 3, 4])

    t2 = Tensor([1., 2, 3, 4])

    opt = SGD([t1, t2], lr=0.1)

    t3 = t1 + t2

    t4 = Tensor([3., 6, 9, 12])

    (t3 - t4).abs().sum(0).backward()

    opt.step()

    print(t1)

    print(t2)


def mul_backprop_test():
    weight = Tensor([[0.], [0.]], True)
    # bias = Tensor([0., ], autograd=True)

    data = Tensor([[2., 2],
                   [3., 3],
                   [4., 7]], autograd=False)
    label = Tensor([[3.], [4], [9]], autograd=False)

    opt = Adam([weight], lr=0.1)

    for i in range(1000):
        loss = (data.mm(weight) - label).abs().sum(0)

        loss.backward()

        opt.step(zero=True)

        print(weight, loss)


def bias_backprop_test():
    bias = Tensor([0.], autograd=True)

    data = Tensor([[2.],
                   [3.],
                   [4.]], autograd=False)

    opt = Adam([bias], lr=0.1)

    for i in range(1000):
        loss = (data + bias).sum(0)

        loss.backward()

        opt.step(zero=True)

        print(bias, loss)


bias_backprop_test()
