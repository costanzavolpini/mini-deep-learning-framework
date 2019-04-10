import torch
from rocket_deepl.module import *


class MSEloss(Module):
    def __call__(self, estimated, target):
        return self.forward(estimated, target)

    #TODO:
    def forward(self, estimated, target):
        self.estimated = estimated
        self.target = target
        return ((estimated - target) ** 2).mean()

    #TODO:
    def backward(self, *noparam):
        return 2 * (self.estimated - self.target)