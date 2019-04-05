from rocket_deepl.module import BinaryOperation
import rocket_deepl.tensor as tensor

class loss(BinaryOperation):

    def __init__(self):
        super().__init__()


    def __call__(self, estimated, target):
        return self.forward(estimated, target)


    #TODO:
    def forward(self, estimated, target):

        super().forward(estimated, target)

        return tensor.Tensor(((estimated.data - target.data) ** 2).mean(), prev_op = self)



    #TODO:
    def backward(self, gradientwrtoutput):

       pass
