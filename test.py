from src.core.nn import Model, Sequential, Linear, Conv2d
from src.data import split
from src.utils import crossEntropyLoss
from src.core.nn import Linear

import numpy as np

from src.utils import SGD, Trainer

m = Linear(2, 1)

data = np.randn(100, 2)

labels = np.randn(100, 1)

train_dataset, test_dataset = split(data, labels, 0.8)

opt = SGD(parameters=m.get_parameters(), alpha=0.01)

trainer = Trainer(m, opt, crossEntropyLoss, {
    "epochs": 10,
    "batch_size": 10,
    "shuffle": True
})

loss = trainer.train(train_dataset, test_dataset)
