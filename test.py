import torch

from generator_training_test import generator
from rocket_deepl.module import *
from rocket_deepl.core.activations.relu import *
from rocket_deepl.core.activations.tanh import *
from rocket_deepl.core.layers import *
from rocket_deepl.sequential import *
from rocket_deepl.utils import *


train_input, train_target = torch.load('train'), torch.load('target')
test_input, test_target = generator(1000)

model = Sequential([
Linear(2, 25), ReLU(),
Linear(25,25), tanH(), 
Linear(25,25), ReLU(), 
Linear(25,25), ReLU(),
Linear(25, 2), tanH()
])

epochs = 200
mini_batch_size = 1

train_model(model, train_input, train_target, epochs, mini_batch_size)


model.plot_accuracy_loss()

number = compute_nb_errors(model, test_input, test_target, mini_batch_size)

accuracy = (1-number/test_input.size(0))*100
number_of_misprediction = number

print("accuracy after train for {} epochs: {}% \n number of miss-predictions : {}".format(epochs, accuracy, number_of_misprediction))

