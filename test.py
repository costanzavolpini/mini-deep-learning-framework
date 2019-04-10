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


train_input, train_target = generator(1000,True)
test_input, test_target = generator(1000,True)


#print(train_target)


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

"""



#RELU test

"""
input_ = torch.ones(4,1)



input_[0,0] = -1.0
input_[1,0] = -2.0


print("input :\n", input_)


activation_relu = ReLU().forward(input_)

print("relu:\n",activation_relu)


activation_tanh= tanH().forward(input_)

print("tanh:\n",activation_tanh)


"""
layer_1= Linear(2,2)
#print(layer_1)


model = Sequential([layer_1])

epochs = 1
mini_batch_size = 1


s = train_input.narrow(0, 0, mini_batch_size)
for e in range(0, epochs):
    for batch in range(0, train_input.size(0), mini_batch_size):

        input = train_input.narrow(0, batch, mini_batch_size)
        target = train_target.narrow(0, batch, mini_batch_size)

        model(input, target)

        print("loss at epoch {} = {}".format(e,model.loss))
        
        model.zero_grad()
        #model.backward()
        #model.step()








