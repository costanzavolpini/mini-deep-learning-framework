import torch
from rocket_deepl.module import *


class  tanH(Module):

    def forward(self, input):
        inp = input.tanh()
        #inp[inp == -1] = 0
        inp = inp.t()
        return inp

    def backward(self, gradientwrtoutput):
        "Derivative of tanh(x) is 1 - tanh^2(x)"
        return 1 - (gradientwrtoutput.tanh() ** 2)