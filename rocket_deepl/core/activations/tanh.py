import torch
from rocket_deepl.module import *


class  tanH(Module):

    def forward(self, input):
        return input.tanh()

    def backward(self, gradientwrtoutput):
        "Derivative of tanh(x) is 1 - tanh^2(x)"
        return 1 - (gradientwrtoutput.tanh() ** 2)