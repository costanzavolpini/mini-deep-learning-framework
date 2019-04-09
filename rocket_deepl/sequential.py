from rocket_deepl.module import Module
import torch

class Sequential(Module):
    def __init__(self, *layers):

        """
        layers is a list of modules which can also be empty.
        """

        self.modules = layers
        self.loss = 0
        

    def forward(self, input):
        pass

    def backward(self, gradientwrtoutput):
        pass

    def fit(self, x_train, 
              target, optimizer = 'SGD',  
              epochs=25,
              cross_val = True, batch_size = 1,
              loss='mse', early_stopping=False,):

        pass

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
        



        




    
    



#TODO: a lot of works! (train and test fn)