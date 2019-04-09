import torch
from rocket_deepl.module import *

class Linear(Module):
    def __init__(self, input_layer, output_layer):
        """
        input_layer = l
        input_layer = l + 1
        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.b = torch.empty((output_layer.shape[0], 1))
        self.w = torch.empty((output_layer.shape[0], input_layer.shape[0]))

        # gradients respect to weight and gradients respect to bias
        # TODO: check shape to be sure
        self.grad_w = torch.empty((self.w.shape))
        self.grad_b = torch.empty((self.b.shape))

    def forward(self, input_layer_before):
        """
        input_layer_before = l - 1
        """
        return self.w @ input_layer_before + self.b

    def backward(self, gradientwrtoutput):
        # gradientwrtoutput = is given and is a future
        grad_w = gradientwrtoutput @ (self.input_layer).t
        self.grad_w += grad_w
        self.grad_b += gradientwrtoutput
        return grad_w


    def param(self):
        return self.w, self.b

    def reset_weights(self):
        #TODO: initialize looking on normal distribution and std
        pass
        # return torch.empty((self.weight.shape))

    def __str__(self):
        return ("Bias: {}\n Weight: {}\n Gradient respect to weight: {}\n Gradient respect to bias: {}\n".format(self.b, self.w, self.grad_w, self.grad_b))


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








