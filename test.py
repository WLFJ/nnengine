from src.core.tensor import Tensor

from src.core.nn import Linear

import numpy as np

model = Linear(10, 3)

data = Tensor(np.random.randn(1, 10))
truth = Tensor(np.array([1, 0, 0]))

d = model.forward(data)

loss = ((d - truth) * (d - truth)).sum(0)

loss.backward()
