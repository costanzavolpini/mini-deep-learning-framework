import torch
from rocket_deepl.module import *

class MSEloss(Module):
    "Class to apply mean square error to detec the loss."

    def __call__(self, estimated, target):
        """
        Sugar function (e.g. MSEloss(0.8, 1) will call the function forward).
        Input:
            estimated: value estimated
            target: value (1 or 0)
        Output:
            mse loss
        """
        return self.forward(estimated, target)


    def forward(self, estimated, target):
        """
        Mean square error: sum (((estimated-target)^2) / #elements)
        Input:
            estimated: value estimated
            target: value (1 or 0)
        Output:
            mse loss
        """
        self.estimated = estimated
        self.target = target

        loss_mse = ((estimated - target) ** 2).mean()
        return loss_mse


    def backward(self, *noparam):
        """
        Derivative of MSE: 2 * (estimated - target)
        Input:
            *noparam: 0 or 1 parameter (in case of the final output there would be zero param) -> then 0 para
        Output:
            gradient of the loss with respect to the input
        """
        to_return =  2 * (self.estimated - self.target)

        return to_return