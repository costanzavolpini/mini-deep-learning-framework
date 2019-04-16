import torch
from rocket_deepl.module import *


class  ReLU(Module):
    "Activation non-linear function: Rectified Linear Units"

    def forward(self, input):
        """
        Apply relu function on input
        Input:
            input: value
        Output:
            input: input where we have applied relu
        """

        input[input < 0 ] = 0.0
        self.input = input

        return  self.input

    def backward(self, gradientwrtoutput):
        """
        Since relu is not differentiable in 0, we just split the two cases:
        where the value > 0 the derivative is 1 else is 0. In 0 is not differentiable!
        Input:
            gradientwrtoutput: gradient respect to the output
        Output:
            gradient of the loss with respect to the input
        """

        self.input[self.input < 0 ]  = 0
        self.input[self.input > 0 ]  = 1

        return gradientwrtoutput * self.input