import torch
from rocket_deepl.module import *


class  tanH(Module):
    """
    Non-linear function used to guarantee a good accuracy of the model.
    Negative inputs are mapped strongly negative and 0 will mapped near 0.
    It is differentiable!
    Range: (-1, 1)

    Used for binary classification.
    """

    def forward(self, input):
        inp = input.tanh()
        #TODO: remove comment of debug
        #inp[inp == -1] = 0

        #TODO: transpose
        #inp = inp.t()
        return inp

    def backward(self, gradientwrtoutput):
        "Derivative of tanh(x) is 1 - tanh^2(x)"
        return 1 - (gradientwrtoutput.tanh() ** 2)