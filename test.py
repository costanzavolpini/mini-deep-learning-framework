import torch
from generator_training_test import generator
from rocket_deepl.module import *
from rocket_deepl.core.activations.relu import *
from rocket_deepl.core.activations.tanh import *
from rocket_deepl.core.layers import *
from rocket_deepl.sequential import *



#from rocket_deepl.core.layers import Linear
#from rocket_deepl.sequential import Sequential 




train_input, train_target = generator(1000)
test_input, test_target = generator(1000)




model = Sequential([ Linear(3,4) , ReLU(),  Linear(3,4), ReLU()])

def train_model(model, train_input, train_target, epochs=25, mini_batch_size = 1):
    # do cross validation and batch inside here
    for e in range(0, epochs):
        for x in range(0, train_input.size(0), mini_batch_size):
            output = model.fit(train_target.narrow(0, x, mini_batch_size))
            model.zero_grad()
            model.backward()

        
            # SGD().update_weights() #TODO: update the weights

            #TODO: output the loss (bonus accuracy viz)


            #TODO: 

#train_model(model, train_input, train_target, 250, 3)






Module()
ReLU()
tanH()
