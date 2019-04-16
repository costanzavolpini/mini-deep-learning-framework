import torch
from rocket_deepl.module import *

from rocket_deepl.core.losses.l_mse import *
from rocket_deepl.optimizer.sgd import *


class Sequential(Module):
    """
    Class that handles different classes. As input it takes a list of layers that compose the neural net.
    """

    def __init__(self, layers, optimizer = 'sgd', loss_layer = MSEloss(), lr=1e-3):
        """
        layers is a list of modules which can also be empty.
        layers = [Linear(), Relu(), Linear()...]
        Input:
            layers: list of classes
            optimizer: optimizer used
            loss_layer: type of loss
            lr: learning rate
        """
        #TODO: handle exception handling
        self.modules = layers
        self.learning_rate = lr
        self.loss = 0

        #TODO: add condition to handle several losses and optimizer!!
        #TODO: add enum for optimizer
        self.loss_layer = loss_layer
        self.optimizer = SGD(self, lr)

        #added loss at last layer
        self.modules.append(loss_layer)



    def __call__(self, x_train,
                 target):
        """
        Sugar function to call the forward.
        Input:
            x_train: train value
            target: target value (0 or 1)
        """
        return self.forward(x_train, target)

    def step(self):
        """
        Optimization that call the update parameters.
        """
        self.optimizer.update_weight()

    def forward(self, input, target):
        """
        Apply forward pass on all layers (excluded loss layer)
        Input:
            input: value
            target: value (0 or 1) the size is 2xbatch_size
        """

        #batch_size x 2 
        self.predicted  = torch.empty(target.size(1))

        for i in range(input.size(0)) :

            inp = input[i].view(-1,2)
            targ = target[:,i].view(2,-1)

            x = inp.view(-1,1)

            #dont take the last layer since it behaves differently
            for l in range(len(self.modules)-1) :
                x = self.modules[l].forward(x)
        
            self.predicted[i] = torch.argmax(x)
            self.loss += self.modules[-1].forward(x,targ)

        return x


    def backward(self):
        """
        Apply backward pass on all the layers starting from the last one
        """
        gradientwrtoutput = []

        #reverse for backward propogation
        self.modules.reverse()

        for layer in self.modules:
            gradientwrtoutput = layer.backward(gradientwrtoutput)

        self.loss = 0

        #reset the loss
        self.modules.reverse()


    def fit(self, x_train,
              target):
        """
        Fit function, call forward pass on all layers.
        """

        print(x_train.size(0))
        for x in range(0, x_train.size(0)):
            self.forward(x,target)

    def compute_accuracy(self):
        #TODO: implement
        pass

    def plot_history(self):
        #TODO: implement
        pass

    def save_model(self, name):
        """
        Function to save the model given the name.
        Input:
            name: name of model to save
        """
        torch.save(self, name)

    def load_model(self, name):
        """
        Function to load the model given the name.
        Input:
            name: name of model to load
        """
        torch.load(name)

    def zero_grad(self):
        """
        Apply zero_grad function on Linear layer
        """
        for l in self.modules:
            if(type(l) is Linear):
                l.zero_grad()
    

    def get_predicted(self):

        return self.predicted