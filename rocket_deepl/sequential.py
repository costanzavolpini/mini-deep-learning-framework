import torch
from rocket_deepl.module import *


class Sequential(Module):

    def __init__(self, layers):
        """
        layers is a list of modules which can also be empty.
        layers = [Linear(), Relu(), Linear()...]
        """
        self.modules = layers
        self.loss = 0
        self.weights = []


    def forward(self, input):
        for layer in self.modules :
            input = layer.forward(input)
        return input #TODO: check if we need or not need to return the input


    def backward(self, gradientwrtoutput):
        #TODO: check reverse and so on!
        for layer in self.modules.reverse():
            gradientwrtoutput = layer.backward(gradientwrtoutput)
            self.weights.append(gradientwrtoutput)
        self.weights.reverse()

    def fit(self, x_train,
              target, optimizer = 'SGD',
              epochs=25,
              loss='mse', early_stopping=False,):

        for e in epochs:
            for x in range(0, x_train.size(0)):
                val = self.forward(x)
                self.backward(val)
                print(self.compute_accuracy)

    def compute_accuracy(self):

        pass


    def compute_nb_miss_predictions(self):

        pass


    def plot_history(self):

        pass


    def save_weights(self):

        pass

    def load_weights(self):

        pass

    def zero_grad(self):
        for l in self.modules:
            if(type(l) is Linear):
                l.zero_grad()
