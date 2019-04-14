import torch
from rocket_deepl.module import *
from rocket_deepl.core.layers import *


class SGD():
    """
    Stochastic gradient descent consists of updating the parameters w_t after every sample w_{t+1}
    w_{t+1} = w_t - step * gradient of loss_{n(t)} (w_t)
    """

    def __init__(self, model, step):
        # step = learning rate
        self.model = model
        self.step = step

    def update_weight(self):
        # self.weight = [ [w_l], [w_l+1], ... , [w_n] ]
        # self.grad_loss = [ [gl_l], [gl_l+1], ... , [gl_n] ]

        for layer in self.model.modules :
            if(type(layer) is Linear):

                layer.w = layer.w - (self.step * layer.grad_w)
                layer.b = layer.b - (self.step * layer.grad_b)


