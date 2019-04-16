import torch
from rocket_deepl.module import *
import math

class Linear(Module):
    """
    Fully connected layer.
    Parameters: weights and biases (initialized by uniform random distribution [-stdv, stdv]).

    Notations:
    input_layer = l
    input_layer = l + 1
    """

    def __init__(self, input_layer, output_layer):
        """
        Initialization of class Linear.
        Input:
            input_layer: numbers of nodes in input layer
            output_layer: numbers of nodes in output layer
        """

        self.stdv = 1. / math.sqrt(output_layer)

        self.input_layer = input_layer
        self.output_layer = output_layer

        # Initialization in a uniform random distribution [-stdv, stdv]
        self.w = torch.empty((output_layer, input_layer)).uniform_(-self.stdv, self.stdv)
        self.b = torch.empty((output_layer, 1)).uniform_(-self.stdv, self.stdv)

        # gradients respect to weight and gradients respect to bias
        self.grad_w = torch.empty((self.w.shape))
        self.grad_w[:,:] = 0.0

        self.grad_b = torch.empty((self.b.shape))
        self.grad_b[:,:] = 0.0


    def forward(self, input_layer_before):
        """
        Notation: input_layer_before = l - 1
        We calculate the output with the formula o = wx + b
        Input:
            input_layer_before: x
        Output:
            output = wx + b
        """
        self.input_layer_before = input_layer_before

        output  = (self.w.mm(input_layer_before)) + self.b

        return output

    def backward(self, gradientwrtoutput):
        """
        Receive the gradient from its OUTPUT (gradientwrtoutput).
        Calculate new gradient respect of the weights.
        Update accumulator fields.
        Input:
            gradientwrtoutput: gradient respect to the output
        Output:
            gradient of the loss with respect to the input
        """
        grad_w = gradientwrtoutput @ self.input_layer_before.t()

        self.grad_w += grad_w
        self.grad_b += gradientwrtoutput

        return self.w.t() @ gradientwrtoutput


    def param(self):
        """
        Return weights and bias
        Output:
            self.w = weights
            self.b = biases
        """
        return self.w, self.b

    def reset_weights(self):
        """
        Resets the weights of the model paramters with
        based on the normal distribution with 0 mean 1e-3 std
        """

        #initialize based on 0 mean and 1e-3 standard deviation
        self.w.uniform_(-self.stdv, self.stdv)
        self.b.uniform_(-self.stdv, self.stdv)


    def __str__(self):
        """
        Method to String
        Output:
            string
        """
        return ("Bias: \n{}\n Weight:\n {}\n Gradient respect to weight:\n {}\n Gradient respect to bias:\n {}\n".format(self.b, self.w, self.grad_w, self.grad_b))

    def zero_grad(self):
        """
        Reset all the gradients respect to weights and biases to zero.
        """
        self.grad_w[:, :] = 0.0
        self.grad_b[:, :] = 0.0

    """
    Functions getter
    """
    def get_input_layer(self):
        """
        Output:
            self.input_layer = #nodes in input layer
        """
        return self.input_layer

    def get_output_layer(self):
        """
        Output:
            self.input_layer = #nodes in output layer
        """
        return self.output_layer

    def get_weight(self):
        """
        Output:
            self.w = weights
        """
        return self.w

    def get_bias(self):
        """
        Output:
            self.b = bias
        """
        return self.b

    def get_grad_weight(self):
        """
        Output:
            self.grad_w = gradient in respect to weight
        """
        return self.grad_w

    def get_grad_bias(self):
        """
        Output:
            self.grad_b = gradient in respect to bias
        """
        return self.grad_b








