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
            value where we have applied tanh
        """
        self.input = input.tanh()

        return self.input

    def backward(self, gradientwrtoutput):
        """
        Derivative of tanh(x) is 1 - tanh^2(x)
        Input:
            gradientwrtoutput: gradient respect to the output
        Output:
            gradient of the loss with respect to the input
        """
        return gradientwrtoutput * (1 - (self.input.tanh() ** 2))