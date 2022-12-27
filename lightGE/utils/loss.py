from lightGE.core.tensor import Tensor

import numpy as np


def mseLoss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) * (pred - target)).sum(0)


def maeLoss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().sum(0)


def crossEntropyLoss(pred: Tensor, target: Tensor) -> Tensor:
    return -target * pred.log() - (Tensor(np.ones_like(pred.data)) - target) * pred.log()


def huberLoss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    if diff.abs().data < 1:
        return (diff * diff).sum(0)
    else:
        return diff.abs().sum(0)
