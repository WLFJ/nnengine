from toynn.core.nn import Model, Sequential, Linear, Conv2d
from toynn.data import split
from toynn.utils import mseLoss, maeLoss, crossEntropyLoss
from toynn.core.nn import Linear
from toynn.utils.scheduler import MultiStepLR, StepLR, Exponential, Cosine

import numpy as np

from toynn.utils import SGD, Trainer

m = Linear(2, 1)

data = np.random.randn(100, 2)

labels = data[:, 0:1] + 10 * data[:, 1:2]

dataset = np.concatenate([data, labels], axis=1)

train_dataset, test_dataset = split(dataset, 0.8)

opt = SGD(parameters=m.get_parameters(), lr=0.01)

sch = MultiStepLR(opt, [10, 20, 30, 40, 50, 60, 70, 80, 90])

trainer = Trainer(m, opt, mseLoss, {
    "epochs": 100,
    "batch_size": 10,
    "shuffle": True
}, sch)

loss = trainer.train(train_dataset, test_dataset)

print(loss)
