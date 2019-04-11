import torch
from rocket_deepl.module import *

class  ReLU(Module):
    """
        Non-linear function used to guarantee a good accuracy of the model.
        ReLU is half rectified (from bottom).
        f(z) = 0 when z < 0 and f(z) = z when z >= 0.
        range: [ 0 to infinity)

        Issue: if z < 0 then f(z) = 0 that causes a bad fit or train of our data.
    """

    def forward(self, input):

        input[input < 0] = 0.0

        #input = input.t()
        return input

    def backward(self, gradientwrtoutput):
        """
        Since ReLU is not differentiable in 0, we just split the two cases:
        where the value > 0 the derivative is 1 else is 0.
        """
        gradientwrtoutput[gradientwrtoutput > 0] = 1
        gradientwrtoutput[gradientwrtoutput < 0] = 0
        return gradientwrtoutput