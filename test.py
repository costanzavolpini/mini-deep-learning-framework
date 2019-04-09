import torch
from rocket_deepl import Module, Linear, Relu, Sequential



train_input, train_target = generator(1000)
test_input, test_target = generator(1000)


model = Sequential([Linear(), Relu(), Linear(), Relu()])

def train_model(model, train_input, train_target, epochs=25, mini_batch_size = 1):
    # do cross validation and batch inside here
    for e in range(0, epochs):
        for x in range(0, train_input.size(0), mini_batch_size):
            output = model.fit(train_target.narrow(0, x, mini_batch_size))
            model.zero_grad()
            model.backward()
            # SGD().update_weights() #TODO:

train_model(model, train_input, train_target, 250, 3)
