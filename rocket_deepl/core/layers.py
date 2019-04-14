import torch
from rocket_deepl.module import *
import math

class Linear(Module):
    def __init__(self, input_layer, output_layer):
        """
        input_layer = l
        input_layer = l + 1
        """

        self.stdv = 1. / math.sqrt(output_layer)
        self.input_layer = input_layer
        self.output_layer = output_layer
        
        self.w = torch.empty((output_layer, input_layer)).uniform_(-self.stdv, self.stdv)
        self.b = torch.empty((output_layer, 1)).uniform_(-self.stdv, self.stdv)


        self.w.fill_(0.01)
        self.b.fill_(0.01)

        # gradients respect to weight and gradients respect to bias
        # TODO: check shape to be sure
        self.grad_w = torch.empty((self.w.shape))
        self.grad_w[:,:] = 0.0

        self.grad_b = torch.empty((self.b.shape))
        self.grad_b[:,:] = 0.0

    def forward(self, input_layer_before):
        """
        input_layer_before = l - 1
        """
        self.input_layer_before = input_layer_before
        output  = (self.w.mm(input_layer_before)) + self.b

        return output

    def backward(self, gradientwrtoutput):
        # gradientwrtoutput = is given and is a future

        self.grad_w += gradientwrtoutput @ self.input_layer_before.t()
        self.grad_b += gradientwrtoutput

        return  self.w.t() @ gradientwrtoutput

    def param(self):
        return self.w, self.b

    def reset_weights(self):

        """
        resets the weights of the model paramters with 
        based on the normal distribution with 0 meand 1e-3 std
        """
        
        std = 1e-3
        mean = 0 
        #initialize based on 0 mean and 1e-3 standard deviation
        self.w.normal_(mean, std)
        self.b.normal_(mean, std)


    def __str__(self):
        return ("Bias: \n{}\n Weight:\n {}\n Gradient respect to weight:\n {}\n Gradient respect to bias:\n {}\n".format(self.b, self.w, self.grad_w, self.grad_b))


    """
    ############### GETTER ###############
    """
    def get_input_layer(self):
        return self.input_layer

    def get_output_layer(self):
        return self.output_layer

    def get_weight(self):
        return self.w

    def get_bias(self):
        return self.b

    def get_grad_weight(self):
        return self.grad_w

    def get_grad_bias(self):
        return self.grad_b

    def zero_grad(self):
        self.grad_w[:, :] = 0.0
        self.grad_b[:, :] = 0.0