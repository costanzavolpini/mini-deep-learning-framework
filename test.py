import torch
from generator_training_test import generator
from rocket_deepl.module import *
from rocket_deepl.core.activations.relu import *
from rocket_deepl.core.activations.tanh import *
from rocket_deepl.core.layers import *
from rocket_deepl.sequential import *
from rocket_deepl.utils import *



#from rocket_deepl.core.layers import Linear
#from rocket_deepl.sequential import Sequential 


train_input, train_target = generator(1000)
test_input, test_target = generator(1000)


model = Sequential([ Linear(3,4) , ReLU(),  Linear(3,4), ReLU()])



epochs  = 25
mini_batch_size = 1

train_model(model,
            train_input, 
            train_target, 
            epochs,
            mini_batch_size)

compute_nb_errors(model, test_input, test_target, mini_batch_size)







