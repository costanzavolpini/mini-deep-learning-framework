import torch
from rocket_deepl.module import *


class  tanH(Module):
    "Activation non-linear function: hyperbolic tangent operation"

    def forward(self, input):
        """
        Apply tanh function on input
        Input:
            input: value
        Output:
            inp: value where we have applied tanh
        """
        inp = input.tanh()
        return inp

    def backward(self, gradientwrtoutput):
        """
        Derivative of tanh(x) is 1 - tanh^2(x)
        Input:
            gradientwrtoutput: gradient respect to the output
        Output:
            gradient of the loss with respect to the input
        """
        return 1 - (gradientwrtoutput.tanh() ** 2)