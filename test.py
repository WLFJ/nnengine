from toynn.core.nn import Model, Sequential, Linear, Conv2d
from toynn.data import split
from toynn.utils import crossEntropyLoss
from toynn.core.nn import Linear

import numpy as np

from toynn.utils import SGD, Trainer

m = Linear(2, 1)

data = np.randn(100, 2)

labels = np.randn(100, 1)

dataset = np.concatenate([data, labels], axis=1)

train_dataset, test_dataset = split(dataset, 0.8)

opt = SGD(parameters=m.get_parameters(), lr=0.01)

trainer = Trainer(m, opt, crossEntropyLoss, {
    "epochs": 10,
    "batch_size": 10,
    "shuffle": True
})

loss = trainer.train(train_dataset, test_dataset)
