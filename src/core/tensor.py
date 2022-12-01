import numpy as np
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set


class Tensor(object):

    def __init__(self, data,
                 autograd: bool = True,
                 creation_op=None):

        self.data = np.array(data)
        self.shape = self.data.shape
        self.autograd = autograd
        self.grad = np.zeros_like(self.data)

        self.creation_op = creation_op
        self.dependents = {}

    def all_children_grads_accounted_for(self):
        for cnt in self.dependents.values():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, origin_id=None):
        if self.autograd:
            if grad is None:
                grad = np.ones_like(self.data)
            else:
                self.grad += grad
                grad = self.grad
                if origin_id is not None:
                    if self.dependents[origin_id] == 0:
                        raise Exception("cannot backprop more than once")
                    self.dependents[origin_id] -= 1

            if self.all_children_grads_accounted_for():
                if self.creation_op is not None:
                    self.creation_op.backward(grad)

    def __add__(self, other):
        op: Op = AddOp(self, other)
        return op.calc()

    def __neg__(self):
        op: Op = NegOp(self)
        return op.calc()

    def __sub__(self, other):
        op: Op = SubOp(self, other)
        return op.calc()

    def __mul__(self, other):
        op: Op = MulOp(self, other)
        return op.calc()

    def mm(self, x):
        op: Op = MatMulOp(self, x)
        return op.calc()

    def exp(self):
        op: Op = ExpOp(self)
        return op.calc()

    def log(self):
        op: Op = LogOp(self)
        return op.calc()

    def sin(self):
        op: Op = SinOp(self)
        return op.calc()

    def cos(self):
        op: Op = CosOp(self)
        return op.calc()

    def sigmoid(self):
        op: Op = SigmoidOp(self)
        return op.calc()

    def tanh(self):
        op: Op = TanhOp(self)
        return op.calc()

    def transpose(self):
        if (self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        return Tensor(self.data.transpose())

    def sum(self, dim):
        op: Op = SumOp(self, dim)
        return op.calc()

    def max(self, axis):
        op: Op = MaxOp(self, axis)
        return op.calc()

    def mean(self, dim):
        op: Op = MeanOp(self, dim)
        return op.calc()

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


# TODO grad_fn 是否支持静态、动态重载
class Op:

    def __init__(self, args):
        self.input: [Tensor] = args
        self.output = None
        self.grad_fn = []

    def calc(self):
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        assert len(self.input) == len(self.grad_fn)

        for i in range(len(self.input)):
            self.input[i].backward(self.grad_fn[i](grad, self.output, self.input), id(self.input[i]))

    def add_dependency(self):
        for i in range(len(self.input)):
            output_id = id(self.output)
            if id(self.output) not in self.input[i].dependents:
                self.input[i].dependents[output_id] = 1
            else:
                self.input[i].dependents[output_id] += 1


class AddOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        super(AddOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * np.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data + self.input[1].data)
        return self.output


class SubOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        super(SubOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * -np.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data - self.input[1].data)
        return self.output


class MulOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        super(MulOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data,
            lambda grad, out, args: grad * args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data * self.input[1].data)
        return self.output


class NegOp(Op):

    def __init__(self, t1: Tensor):
        super(NegOp, self).__init__([t1])
        self.grad_fn = [
            lambda grad, out, args: grad * -np.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(-self.input[0].data)
        return self.output


class MatMulOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        super(MatMulOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad @ args[1].data.transpose(),
            lambda grad, out, args: args[0].data.transpose() @ grad
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.dot(self.input[1].data))
        return self.output


class ExpOp(Op):
    def __init__(self, t: Tensor):
        super(ExpOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * out.data
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.exp(self.input[0].data))
        return self.output


class LogOp(Op):
    def __init__(self, t: Tensor):
        super(LogOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad / args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.log(self.input[0].data))
        return self.output


class SinOp(Op):
    def __init__(self, t: Tensor):
        super(SinOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * np.cos(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.sin(self.input[0].data))
        return self.output


class CosOp(Op):
    def __init__(self, t: Tensor):
        super(CosOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * -np.sin(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.cos(self.input[0].data))
        return self.output


class SigmoidOp(Op):

    def __init__(self, t: Tensor):
        super(SigmoidOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * out.data * (1 - out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(1 / (1 + np.exp(-self.input[0].data)))
        return self.output


class TanhOp(Op):

    def __init__(self, t: Tensor):
        super(TanhOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * (1 - out.data * out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.tanh(self.input[0].data))
        return self.output


class TransposeOp(Op):

    def __init__(self, t: Tensor, axes: Iterable[int] = None):
        super(TransposeOp, self).__init__([t])
        self.axes = axes

        self.grad_fn = [
            lambda grad, out, args: grad.transpose(self.axes)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.transpose(self.axes))
        return self.output


class SumOp(Op):

    def __init__(self, t: Tensor, dim: int):
        super(SumOp, self).__init__([t])
        assert dim < len(t.data.shape)
        self.dim = dim
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.dim))
        return self.output


class MaxOp(Op):

    def __init__(self, t: Tensor, dim: int):
        super(MaxOp, self).__init__([t])
        assert dim < len(t.data.shape)
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data)
        ]
        self.dim = dim
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.max(axis=self.dim))
        return self.output


class MeanOp(Op):

    def __init__(self, t: Tensor, dim: int):
        super(MeanOp, self).__init__([t])
        assert dim < len(t.data.shape)
        self.dim = dim
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.mean(axis=self.dim))
        return self.output


def log(t: Tensor) -> Tensor:
    return t.log()


def exp(t: Tensor) -> Tensor:
    return t.exp()


def sin(t: Tensor) -> Tensor:
    return t.sin()


def cos(t: Tensor) -> Tensor:
    return t.cos()


def tanh(t: Tensor) -> Tensor:
    return t.tanh()


def sigmoid(t: Tensor) -> Tensor:
    return t.sigmoid()


def mm(t1: Tensor, t2: Tensor) -> Tensor:
    return t1.mm(t2)
