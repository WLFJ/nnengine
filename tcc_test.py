'''
TcGraph, a part of T-Lang.

TcGraph aimed to express tensor computation in graph form,
so that it'll be converted into T-Lang for further
optimization.

Currently we simply tracking every tensor created, so for
training, it cannot recognize there's some same pattern in
it, the result it cause will make TcGraph very big.
'''

from lightGE.core import Tensor
from lightGE.core import TcGraph

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = Tensor([100, 100, 100])

d = a + b + c

print(TcGraph.get_instantce().tmap)
print(TcGraph.get_instantce().graph)

print(TcGraph.Compile())

TcGraph.Clear()

print(d)
