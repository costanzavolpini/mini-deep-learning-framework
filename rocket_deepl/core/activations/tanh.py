from rocket_deepl.module import UnaryOperation
import rocket_deepl.tensor as tensor
import tanh


class  tanH(UnaryOperation):


    #TODO:
    def forward(self,input):
        
        super().forward(input)

        return tensor.Tensor(input.data.tanh(), prev_op = self)
    
        

    #TODO:
    def backward(self, gradientwrtoutput):
    
       pass
