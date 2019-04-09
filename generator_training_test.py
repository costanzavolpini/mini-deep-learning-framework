import torch
import math



def generator(n):
    input = torch.Tensor(n, 2).uniform_(0, 1) #[0,1]^2
    radius = 1/(2*math.pi) #0.398...
    target = (input - 0.5).pow(2).sum(1).sub(radius).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generator(1000)
test_input, test_target = generator(1000)