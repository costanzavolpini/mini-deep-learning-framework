import torch
from rocket_deepl.module import *




class  ReLU(Module):

    def forward(self, input):
        return input.relu()

    #TODO:
    def backward(self, gradientwrtoutput):
        """
        Since relu is not differentiable in 0, we just split the two cases:
        where the value > 0 the derivative is 1 else is 0. In 0 is not differentiable!
        """
        gradientwrtoutput[gradientwrtoutput > 0] = 1
        gradientwrtoutput[gradientwrtoutput < 0] = 0
        return gradientwrtoutput