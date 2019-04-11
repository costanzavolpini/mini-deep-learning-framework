import torch
from rocket_deepl.module import *


class MSEloss(Module):
    """
    TODO:
    """
    def __call__(self, estimated, target):
        return self.forward(estimated, target)

    #TODO:
    def forward(self, estimated, target):
        self.estimated = estimated
        self.target = target
   


        loss_mse = ((estimated - target) ** 2).mean()



        return loss_mse


    #TODO:
    def backward(self, *noparam):
        to_return =  2* (self.estimated - self.target)


        return to_return