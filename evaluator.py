import tensor

from nn import Layer


class Evaluator(object):
    def __init__(self):
        self.loss = 0
        self.num_correct = 0
