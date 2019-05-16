import torch
from rocket_deepl.module import *
import matplotlib.pyplot as plt
from rocket_deepl.core.losses.l_mse import *
from rocket_deepl.optimizer.sgd import *

class Sequential(Module):
    """
    Class that handles different modules. As input it takes a list of layers that composes the neural net.
    """

    def __init__(self, layers, loss_layer=MSEloss(), lr=1e-3):
        """
        Function to inizialize.
        As optimizer we have used SGD, as loss we have used MSE.
        Input:
            layers: list of modules which can also be empty. (layers = [Linear(), Relu(), Linear()...])
            lr: learning rate
        """
        self.modules = layers
        self.learning_rate = lr
        self.loss = 0

        self.loss_layer = loss_layer
        self.optimizer = SGD(self, lr)

        # added loss at last layer
        self.modules.append(loss_layer)

        self.plot_loss = []
        self.plot_accuracy = []


    def __call__(self, x_train, target):
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
        Apply forward pass on all layers (excluded loss layer).
        Input:
            input: value
            target: value (0 or 1) -> shape (2 x batch_size)
        """
        self.predicted  = torch.empty(target.size(1))

        for i in range(input.size(0)):
            inp = input[i].view(-1, 2)
            targ = target[:, i].view(2, -1)

            x = inp.view(-1, 1)

            # do not take the last layer since it behaves differently (loss layer)
            for l in range(len(self.modules) - 1):
                x = self.modules[l].forward(x)

            # retrive max index of the the target
            self.predicted[i] = torch.argmax(x)
            self.loss += self.modules[-1].forward(x, targ)
        return x


    def backward(self):
        """
        Apply backward pass on all the layers starting from the last one.
        """
        gradientwrtoutput = []

        # reverse modules list for backward propogation
        self.modules.reverse()

        for layer in self.modules:
            gradientwrtoutput = layer.backward(gradientwrtoutput)

        self.loss = 0 # reset the loss after the backward

        # reverse modules to original list
        self.modules.reverse()


    def fit(self, x_train, target):
        """
        Fit function, call forward pass on all layers.
        """
        for x in range(0, x_train.size(0)):
            self.forward(x, target)


    def plot_accuracy_loss(self):
        """
        Function to plot the accuracy and loss.
        """
        plt.plot(self.plot_accuracy, color='indianred', label="accuracy")
        plt.plot(self.plot_loss, color='royalblue', label="loss")

        plt.xlabel("number of epochs")
        plt.legend(loc ='upper left')
        plt.title("Plot that shows accuracy and loss on {} epochs".format(len(self.plot_loss)))

        # save the file
        plt.savefig("accuracy_loss.png")
        plt.show()


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
        Apply zero_grad function (set gradients to zero)  on linear layers.
        """
        for l in self.modules:
            if(type(l) is Linear):
                l.zero_grad()


    def get_predicted(self):
        """
        Get the predicted value.
        """
        return self.predicted