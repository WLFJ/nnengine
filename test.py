import torch

from lightGE.core.nn import Model, Sequential, Linear, Conv2d
from lightGE.data import Dataset
from lightGE.utils import mseLoss, maeLoss, crossEntropyLoss
from lightGE.core.nn import Linear
from lightGE.utils.scheduler import MultiStepLR, StepLR, Exponential, Cosine

import numpy as np

from lightGE.utils import SGD, Trainer

import logging

logging.basicConfig(level=logging.INFO)

m = Linear(2, 1)

data = np.random.randn(100, 2)

labels = data[:, 0:1] + 10 * data[:, 1:2]

dataset = Dataset(data, labels)

train_dataset, test_dataset = dataset.split(0.8)

opt = SGD(parameters=m.params(), lr=0.01)

sch = MultiStepLR(opt, [10, 20, 30, 40, 50, 60, 70, 80, 90])

trainer = Trainer(m, opt, mseLoss, {
    "epochs": 100,
    "batch_size": 10,
    "shuffle": True,
    "save_path": "./tmp/model.pkl"
}, sch)

loss = trainer.train(train_dataset, test_dataset)

print(loss)
