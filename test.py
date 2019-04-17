import torch

from generator_training_test import generator
from rocket_deepl.module import *
from rocket_deepl.core.activations.relu import *
from rocket_deepl.core.activations.tanh import *
from rocket_deepl.core.layers import *
from rocket_deepl.sequential import *
from rocket_deepl.utils import *


# Decomment to load the dataset
# train_input, train_target = torch.load('train'), torch.load('target')
train_input, train_target = generator(1000)

# Generate the dataset with one-hot encoding
test_input, test_target = generator(1000)

# Model with 2 input units, 2 output units and 3 hidden layers
model = Sequential([
Linear(2, 25), ReLU(),
Linear(25,25), ReLU(),
Linear(25,25), ReLU(),
Linear(25, 2), tanH()
])

epochs = 50
mini_batch_size = 1

# train
train_model(model, train_input, train_target, epochs, mini_batch_size)

# plot accuracy and loss
model.plot_accuracy_loss()

# compute miss prediction
number = compute_nb_errors(model, test_input, test_target, mini_batch_size)
number_of_misprediction = number


# compute accuray
accuracy = (1 - number/test_input.size(0))*100

print("accuracy after train for {} epochs: {}% \n number of miss-predictions : {}".format(epochs, accuracy, number_of_misprediction))

