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

        input[input < 0 ] = 0.0
        self.input = input

        return  self.input

    #TODO:
    def backward(self, gradientwrtoutput):
        """
        Since ReLU is not differentiable in 0, we just split the two cases:
        where the value > 0 the derivative is 1 else is 0.
        """

        #print(self.input.shape)

        self.input[self.input < 0 ]  = 0
        self.input[self.input > 0 ]  = 1
         
       # print(self.input.shape)
       # print(gradientwrtoutput.shape)
        output = gradientwrtoutput * self.input

        #print(output.shape)
        return output