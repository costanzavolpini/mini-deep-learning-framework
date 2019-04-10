import torch
from rocket_deepl.module import *

from rocket_deepl.core.losses.l_mse import *
from rocket_deepl.optimizer.sgd import *


class Sequential(Module):

    def __init__(self, layers, optimizer = 'sgd', loss_layer = MSEloss(), lr=0.01):
        """
        layers is a list of modules which can also be empty.
        layers = [Linear(), Relu(), Linear()...]
        """

        #TODO: handle exception handling
        self.modules = layers
        self.learning_rate = lr

        self.loss = 0

        #TODO: add condition to handle several losses
        self.loss_layer = loss_layer
        self.optimizer = SGD(self, lr)

        #added loss at last layer
        self.modules.append(loss_layer)

    def __call__(self, x_train, 
                 target):
        self.fit(x_train, target)
        
    def step(self):
      self.optimizer.update_weight()

    def forward(self, input, target):

        print("input :---->", input)
        print("target:---->", target)

        #dont take the last layer since it behaves differently
        for l in range(len(self.modules)-1) :
            
            input = self.modules[l].forward(input)

        #get mse layer and apply the target


        print(target)



        self.loss += self.modules[-1].forward(input,target)




    def backward(self):
        gradientwrtoutput = []
        #reverse for backward propogation
        for layer in self.modules.reverse():
            gradientwrtoutput = layer.backward(gradientwrtoutput)

        #reset the loss 
        self.loss = 0

    def fit(self, x_train,
              target):

        for x in range(0, x_train.size(0)):
            self.forward(x,target)

    def compute_accuracy(self):
        pass

    def plot_history(self):
        pass

    def save_model(self,name):
        torch.save(self, name)

    def load_model(self, name):
        torch.load(name)

    def zero_grad(self):
        for l in self.modules:
            if(type(l) is Linear):
                l.zero_grad()
