from toynn.core.tensor import Tensor

import numpy as np


def mseLoss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) * (pred - target)).sum(0)


def maeLoss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().sum(0)


def crossEntropyLoss(pred: Tensor, target: Tensor) -> Tensor:
    return -target * pred.log() - (Tensor(np.ones_like(pred.data)) - target) * pred.log()