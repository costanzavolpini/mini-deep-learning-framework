import torch
from rocket_deepl.module import *

# Stochastic gradient descent consists of updating the parameters w_t after every sample w_{t+1}
# w_{t+1} = w_t - step * gradient of loss_{n(t)} (w_t)

class SGD():

    def __init__(self, weight, grad_loss, step):
        # step = learning rate
        self.weight = weight
        self.grad_loss = grad_loss
        self.step = step


    def update_weight(self):
        # self.weight = [ [w_l], [w_l+1], ... , [w_n] ]
        # self.grad_loss = [ [gl_l], [gl_l+1], ... , [gl_n] ]
        return self.weight - (self.step * self.grad_loss)

