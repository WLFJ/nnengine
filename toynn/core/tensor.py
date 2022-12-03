import numpy as np
import scipy.special


class Tensor(object):

    def __init__(self, data,
                 autograd: bool = True,  # TODO 这里应该是 False
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

    def __truediv__(self, other):
        op: Op = DivOp(self, other)
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

    def relu(self):
        op: Op = ReLuOp(self)
        return op.calc()

    def softmax(self):
        op: Op = SoftmaxOp(self)
        return op.calc()

    def transpose(self):
        op: Op = TransposeOp(self)
        return op.calc()

    def abs(self):
        op: Op = AbsOp(self)
        return op.calc()

    def sum(self, dim):
        op: Op = SumOp(self, dim)
        return op.calc()

    def max(self, axis):
        op: Op = MaxOp(self, axis)
        return op.calc()

    def mean(self, dim):
        op: Op = MeanOp(self, dim)
        return op.calc()

    def broadcast(self, other):
        if self.shape == other.shape:
            return self, other

        s1 = list(self.shape)
        s2 = list(other.shape)
        if len(s1) > len(s2):
            s2 = [1] * (len(s1) - len(s2)) + s2
            if s1 == s2:
                t = BroadcastOp(other, self.shape).calc()
                return t, other
        else:
            s1 = [1] * (len(s2) - len(s1)) + s1
            if s1 == s2:
                t = BroadcastOp(self, other.shape).calc()
                return self, t

        s = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                if s1[i] == 1:
                    s.append(s2[i])
                elif s2[i] == 1:
                    s.append(s1[i])
                else:
                    raise Exception("cannot broadcast")
            else:
                s.append(s1[i])

        if s != list(self.shape):
            t1 = BroadcastOp(self, s).calc()
        else:
            t1 = self

        if s != list(other.shape):
            t2 = BroadcastOp(other, s).calc()
        else:
            t2 = other
        return t1, t2

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
            self.input[i].backward(self.grad_fn[i](grad, self.output, self.input), id(self.output))

    def add_dependency(self):
        for i in range(len(self.input)):
            output_id = id(self.output)
            if id(self.output) not in self.input[i].dependents:
                self.input[i].dependents[output_id] = 1
            else:
                self.input[i].dependents[output_id] += 1


class AddOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(AddOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * np.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data + self.input[1].data, creation_op=self)
        return self.output


class SubOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(SubOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * -np.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data - self.input[1].data, creation_op=self)
        return self.output


class MulOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(MulOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data,
            lambda grad, out, args: grad * args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data * self.input[1].data, creation_op=self)
        return self.output


class DivOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(DivOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad / args[1].data,
            lambda grad, out, args: grad * -args[0].data / (args[1].data * args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data / self.input[1].data, creation_op=self)
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
            self.output: Tensor = Tensor(-self.input[0].data, creation_op=self)
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
            self.output: Tensor = Tensor(self.input[0].data.dot(self.input[1].data), creation_op=self)
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
            self.output: Tensor = Tensor(np.exp(self.input[0].data), creation_op=self)
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
            self.output: Tensor = Tensor(np.log(self.input[0].data), creation_op=self)
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
            self.output: Tensor = Tensor(np.sin(self.input[0].data), creation_op=self)
        return self.output


class CosOp(Op):
    def __init__(self, t: Tensor):
        super(CosOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * -np.sin(args[0].data, creation_op=self)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.cos(self.input[0].data), creation_op=self)
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
            self.output: Tensor = Tensor(1 / (1 + np.exp(-self.input[0].data)), creation_op=self)
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
            self.output: Tensor = Tensor(np.tanh(self.input[0].data), creation_op=self)
        return self.output


class ReLuOp(Op):
    def __init__(self, t: Tensor):
        super(ReLuOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * (args[0].data > 0)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.maximum(self.input[0].data, 0), creation_op=self)
        return self.output


class TransposeOp(Op):

    def __init__(self, t: Tensor, axes: [int] = None):
        super(TransposeOp, self).__init__([t])
        self.axes = axes
        self.grad_fn = [
            lambda grad, out, args: grad.transpose(
                list(range(len(axes.shape))).sort(key=lambda x: self.axes[x])
            )
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.transpose(self.axes), creation_op=self)
        return self.output


class AbsOp(Op):
    def __init__(self, t: Tensor):
        super(AbsOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * np.sign(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.abs(self.input[0].data), creation_op=self)
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
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.dim), creation_op=self)
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
            self.output: Tensor = Tensor(self.input[0].data.max(axis=self.dim), creation_op=self)
        return self.output


class MeanOp(Op):

    def __init__(self, t: Tensor, dim: int):
        super(MeanOp, self).__init__([t])
        assert dim < len(t.data.shape)
        self.dim = dim
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data) / args[0].data.shape[self.dim]
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.mean(axis=self.dim), creation_op=self)
        return self.output


# TODO softmax 可以指定维度
class SoftmaxOp(Op):

    def __init__(self, t: Tensor):
        super(SoftmaxOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: out.data * (grad - np.sum(grad * out.data, axis=-1, keepdims=True))
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(scipy.special.softmax(self.input[0].data, axis=-1), creation_op=self)
        return self.output


class BroadcastOp(Op):
    def __init__(self, t: Tensor, shape: [int]):
        super(BroadcastOp, self).__init__([t])
        self.shape = shape
        self.axes = []
        if len(shape) > len(t.shape):
            self.axes = list(range(len(shape) - len(t.shape)))

        offset = len(shape) - len(t.shape)
        for i in range(len(t.shape)):
            if t.shape[i] != shape[i + offset]:
                self.axes.append(i + offset)

        self.axes = tuple(self.axes)
        self.grad_fn = [
            lambda grad, out, args: grad.sum(axis=self.axes)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.broadcast_to(self.input[0].data, self.shape), creation_op=self)
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


def relu(t: Tensor) -> Tensor:
    return t.relu()


def mm(t1: Tensor, t2: Tensor) -> Tensor:
    return t1.mm(t2)


def softmax(t: Tensor) -> Tensor:
    return t.softmax()


def abs(t: Tensor) -> Tensor:
    return t.abs()


def sum(t: Tensor, dim: int) -> Tensor:
    return t.sum(dim)


def max(t: Tensor, dim: int) -> Tensor:
    return t.max(dim)


def mean(t: Tensor, dim: int) -> Tensor:
    return t.mean(dim)


def transpose(t: Tensor, axes: [int] = None) -> Tensor:
    return t.transpose(axes)
