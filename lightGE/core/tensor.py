import numpy as np
import scipy.special
from typing import Iterable, List


class Tensor(object):

    def __init__(self, data,
                 autograd: bool = False,
                 creation_op=None):

        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape
        self.autograd = autograd
        if autograd:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        self.creation_op = creation_op
        self.dependents = {}

        self.tcg_id = TcGraph.AddTensor(self)

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

    def __pow__(self, power):
        if not isinstance(power, Tensor):
            power = Tensor(power, autograd=False)
        op: Op = PowOp(self, power)
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

    def abs(self):
        op: Op = AbsOp(self)
        return op.calc()

    def sum(self, axes):
        op: Op = SumOp(self, axes)
        return op.calc()

    def max(self, axes):
        op: Op = MaxOp(self, axes)
        return op.calc()

    def mean(self, axes):
        op: Op = MeanOp(self, axes)
        return op.calc()

    def var(self, axes):
        op: Op = VarOp(self, axes)
        return op.calc()

    def sqrt(self):
        op: Op = SqrtOp(self)
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

    def squeeze(self, dim):
        op: Op = SqueezeOp(self, dim)
        return op.calc()

    def unsqueeze(self, dim):
        op: Op = UnsqueezeOp(self, dim)
        return op.calc()

    def transpose(self, axes: Iterable[int] = None):
        op: Op = TransposeOp(self, axes)
        return op.calc()

    def reshape(self, shape):
        op: Op = ReshapeOp(self, shape)
        return op.calc()

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class TcGraph:
    instance = None

    def __init__(self):
        self.tmap = dict()
        # (op_name, (input1, input2, ...), (output1, output2, ...))
        self.graph = list()

    @classmethod
    def get_instantce(cls):
        if not cls.instance:
            cls.instance = TcGraph()

        return cls.instance

    @classmethod
    def GetTensor(cls, t):
        return cls.get_instantce().getTensor(t)

    @classmethod
    def Compile(cls):
        return cls.get_instantce().compile()

    @classmethod
    def Clear(cls):
        return cls.get_instantce().clear()

    def compile(self):
        '''
        Convert TcGraph into T-Lang program.
        '''

        graph = self.graph
        tensor_dict = dict(map(reversed, self.tmap.items()))

        op_list = ['def main(){\n']

        tensor_input = set()
        tensor_mid = set()

        for (_, __, out) in graph:
            tensor_mid.update(out)

        for (_, inp, __) in graph:
            tensor_input.update(set(inp).difference(tensor_mid))

        # create all input tensor
        for id in tensor_input:
            t = tensor_dict[id]
            shape = 'x'.join(str(d) for d in t.data.shape)
            data = ', '.join(str(e) for e in t.data.flat)
            op = f'  var v{id}<{shape}> = [{data}];\n'
            op_list.append(op)

        while True:
            # if graph not empty, find all which input all generated.
            is_emitable = False
            for (name, inp, out) in graph:
                assert len(out) == 1 and "for now only support 1 result."
                out = out[0]

                is_emitable = out not in tensor_input and set(inp).issubset(tensor_input)
                params = ', '.join(f'v{tid}' for tid in inp)

                if is_emitable:
                    op = f'  var v{out} = {name}({params});\n'
                    if name in ['add', 'matmul']:
                        assert len(inp) == 2 and 'binop must have 2 op.'
                        # TODO: Add more.
                        binop_dict = {
                            'add': '+',
                            'matmul': '.',
                        }
                        op = f'  var v{out} = v{inp[0]} {binop_dict[name]} v{inp[1]};\n'

                    op_list.append(op)

                    tensor_input.add(out)
                    tensor_mid.remove(out)

                    # if cur op's result is the last, also emit printOp.
                    if len(tensor_mid) == 0:
                        op_list.append(f'  print(v{out});\n')
            if not is_emitable:
                break

        op_list.append('}\n')
        return ''.join(op_list)

    def clear(self):
        pass

    def getTensor(self, t):
        '''
        return tensor internal repr id.
        '''
        tmap = self.tmap
        assert type(t) == Tensor and "getTensor input only suppor Tensor."
        if t not in tmap:
            print('TcGraph: Warning: current tensor not managed.')
            return self.addTensor(t)

        # assert t.get_tcg_id() == tmap[t] and "tcg_id and id managed in TcGraph must be the same."
        return tmap[t]

    @classmethod
    def AddTensor(cls, t):
        return cls.get_instantce().addTensor(t)

    def addTensor(self, t):
        '''
        alloc a internal repr id for given tensor.
        '''
        tmap = self.tmap
        if t not in tmap:
            tmap[t] = len(tmap)
        return self.getTensor(t)

    @classmethod
    def AddOp(cls, op_name, inputs, outputs):
        return cls.get_instantce().addOp(op_name, inputs, outputs)

    def addOp(self, op_name, inputs, outputs):
        self.graph.append((op_name,
                           tuple(self.getTensor(i) for i in inputs),
                           tuple(self.addTensor(o) for o in outputs)
                           ))


# TODO grad_fn 是否支持静态、动态重载
class Op:

    def __init__(self, args, tcc_opname='unsupported'):
        self.input: List[Tensor] = args
        self.output: [Tensor, None] = None
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
            self.output: Tensor = Tensor(self.input[0].data + self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('add', [self.input[0], self.input[1]], [self.output])
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
            self.output: Tensor = Tensor(self.input[0].data - self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sub', [self.input[0], self.input[1]], [self.output])
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
            self.output: Tensor = Tensor(self.input[0].data * self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('mul', [self.input[0], self.input[1]], [self.output])
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
            self.output: Tensor = Tensor(self.input[0].data / self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('div', [self.input[0], self.input[1]], [self.output])
        return self.output


class PowOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(PowOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data * np.power(args[0].data, args[1].data - 1),
            lambda grad, out, args: grad * np.log(args[0].data) * np.power(args[0].data, args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(np.power(self.input[0].data, self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('pow', [self.input[0], self.input[1]], [self.output])
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
            self.output: Tensor = Tensor(-self.input[0].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('neg', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(self.input[0].data.dot(self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('matmul', [self.input[0], self.input[1]], [self.output])
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
            self.output: Tensor = Tensor(np.exp(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('exp', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(np.log(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('log', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(np.sin(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sin', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(np.cos(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('cos', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(1 / (1 + np.exp(-self.input[0].data)), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sigmoid', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(np.tanh(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('tanh', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(np.maximum(self.input[0].data, 0), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('relu', [self.input[0]], [self.output])
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
            self.output: Tensor = Tensor(np.abs(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('abs', [self.input[0]], [self.output])
        return self.output


class SumOp(Op):

    def __init__(self, t: Tensor, axes: int):
        super(SumOp, self).__init__([t])
        self.axes = axes
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sum', [self.input[0]], [self.output])
        return self.output


class MaxOp(Op):

    def __init__(self, t: Tensor, axes: int):
        super(MaxOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * (args[0].data == out.data)
        ]
        self.axes = axes
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.max(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('max', [self.input[0]], [self.output])
        return self.output


class MeanOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable]):
        super(MeanOp, self).__init__([t])

        if isinstance(axes, int):
            axes = [axes]

        axes = list(axes)

        for i in range(len(axes)):
            if axes[i] < 0:
                axes[i] += len(t.data.shape)

        self.axes = tuple(axes)

        self._axes = list(t.shape)
        for i in range(len(self._axes)):
            if i in axes:
                self._axes[i] = 1
        self._axes = tuple(self._axes)

        self.N = 1
        for axis in axes:
            self.N *= t.shape[axis]

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._axes) * np.ones_like(args[0].data) / self.N
        ]
        self.calc()
        self.add_dependency()

    # TODO: TcGraph: argument config is needed.
    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes) / self.N,
                                         creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
        return self.output


class VarOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable]):
        super(VarOp, self).__init__([t])

        if isinstance(axes, int):
            axes = [axes]

        axes = list(axes)

        for i in range(len(axes)):
            if axes[i] < 0:
                axes[i] += len(t.data.shape)

        self.axes = tuple(axes)

        self._axes = list(t.shape)
        for i in range(len(self._axes)):
            if i in axes:
                self._axes[i] = 1
        self._axes = tuple(self._axes)

        self.N = 1
        for axis in axes:
            self.N *= t.shape[axis]

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._axes) *
                                    2 * (args[0].data - args[0].data.sum(self.axes, keepdims=True) / self.N) / self.N
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            data = self.input[0].data
            mean_val = data.sum(axis=self.axes, keepdims=True) / self.N
            data = data - mean_val
            data = data * data
            self.output: Tensor = Tensor(data.sum(axis=self.axes) / self.N, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))

        return self.output


class SqrtOp(Op):
    def __init__(self, t: Tensor):
        super(SqrtOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * 0.5 * (1 / out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.sqrt(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
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

    # TODO: TcGraph: argument config is needed.
    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(scipy.special.softmax(self.input[0].data, axis=-1), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('softmax', [self.input[0]], [self.output])
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
            lambda grad, out, args: grad.sum(axis=self.axes).reshape(args[0].shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.broadcast_to(self.input[0].data, self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('broadcast', [self.input[0]], [self.output])
        return self.output


class SqueezeOp(Op):
    def __init__(self, t: Tensor, axis: int):
        super(SqueezeOp, self).__init__([t])
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.squeeze(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('squeeze', [self.input[0]], [self.output])
        return self.output


class UnsqueezeOp(Op):
    def __init__(self, t: Tensor, axis: int):
        super(UnsqueezeOp, self).__init__([t])
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.expand_dims(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('unsqueeze', [self.input[0]], [self.output])
        return self.output


class TransposeOp(Op):

    def __init__(self, t: Tensor, axes: Iterable[int] = None):
        super(TransposeOp, self).__init__([t])
        if axes is None:
            self.axes = list(range(len(t.shape) - 1, -1, -1))
        else:
            self.axes = axes
        self.grad_fn = [
            lambda grad, out, args: grad.transpose(
                list(range(len(self.axes))).sort(key=lambda x: self.axes[x])
            )
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.transpose(self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('transpose', [self.input[0]], [self.output])
        return self.output


class ReshapeOp(Op):
    def __init__(self, t: Tensor, shape: [int]):
        super(ReshapeOp, self).__init__([t])
        self.shape = shape
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.reshape(self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('reshape', [self.input[0]], [self.output])
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


def mean(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.mean(axes)


def var(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.var(axes)


def sqrt(t: Tensor) -> Tensor:
    return t.sqrt()
