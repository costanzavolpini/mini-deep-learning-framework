import torch
import math

def generator(n):
    """
    Function to generate train and target.
    Input:
        n: number of values of training and target
        one_hot_enc: boolean to set one hot encoding
    Output:
        train values and target values
    """
    input = torch.empty(n, 2)
    input = input.uniform_(0, 1) #[0,1]^2
    radius = 1/(2*math.pi) #0.398...
    target = (input - 0.5).pow(2).sum(1).sub(radius).sign().add(1).div(2).long()

    # Enable hot encoding

    # target in this case is an id than if target = 0,
    # put 1 in the first column, if target = 1 we put 1 in second column
    target_hot = torch.empty(target.shape[0], 2)
    target_hot[:, :] = 0 # fill all with 0s

    target_hot[:,0] = target == 0
    target_hot[:,1] = target == 1
    
    return input, target_hot



