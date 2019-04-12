import torch
from rocket_deepl.module import *


class  ReLU(Module):

    def forward(self, input):

        input[input < 0 ] = 0.0
        self.input = input

        return  self.input

    #TODO:
    def backward(self, gradientwrtoutput):
        """
        Since relu is not differentiable in 0, we just split the two cases:
        where the value > 0 the derivative is 1 else is 0. In 0 is not differentiable!
        """

        #print(self.input.shape)

        self.input[self.input < 0 ]  = 0
        self.input[self.input > 0 ]  = 1
         
       # print(self.input.shape)
       # print(gradientwrtoutput.shape)
        output = gradientwrtoutput * self.input

        #print(output.shape)
        return output