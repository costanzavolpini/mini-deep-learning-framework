import torch
from generator_training_test import generator
from rocket_deepl.module import *
from rocket_deepl.core.activations.relu import *
from rocket_deepl.core.activations.tanh import *
from rocket_deepl.core.layers import *
from rocket_deepl.sequential import *
from rocket_deepl.utils import *


train_input, train_target = torch.load("train"), torch.load("target")

#train_input[:,0] = 0.4557
#train_input[:,1] = 0.5492

#train_target[:,0] = 1.0
#train_target[:,1] = 0.

print("training")
print(train_input)
print(train_target)


#model = Sequential([ Linear(3,4) , ReLU(),  Linear(3,4), ReLU()])


epochs  = 25
mini_batch_size = 1

"""
train_model(model,
            train_input,
            train_target,
            epochs,
            mini_batch_size)

compute_nb_errors(model, test_input, test_target, mini_batch_size)

input_ = torch.ones(4,1)



input_[0,0] = -1.0
input_[1,0] = -2.0


print("input :\n", input_)


activation_relu = ReLU().forward(input_)

print("relu:\n",activation_relu)


activation_tanh= tanH().forward(input_)

print("tanh:\n",activation_tanh)

"""


# model = Sequential([Linear(2,2), ReLU()])

model = Sequential([Linear(2, 25), ReLU(),  Linear(25, 25),  ReLU(), Linear(2, 2), tanH()])
#loss : 0.184489905834198

epochs = 1000
mini_batch_size = 1


s = train_input.narrow(0, 0, mini_batch_size)
for e in range(0, epochs):

    model.zero_grad()
    for batch in range(0, train_input.size(0), mini_batch_size):

        input = train_input.narrow(0, batch, mini_batch_size)
        target = train_target.narrow(0, batch, mini_batch_size)

        model(input, target.t())
        model.zero_grad()
        loss = model.loss
        model.backward()
        model.step()


    print("epoch : {}, loss : {}".format(e,loss))
