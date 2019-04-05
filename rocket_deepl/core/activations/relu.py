from rocket_deepl.module import UnaryOperation
import rocket_deepl.tensor as tensor


class  ReLU(UnaryOperation):


    def forward(self,input):
        
        super().forward(input)
        
        return tensor.Tensor(input.data.relu(), prev_op = self)

        


    #TODO:
    def backward(self, gradientwrtoutput):
    
       pass


