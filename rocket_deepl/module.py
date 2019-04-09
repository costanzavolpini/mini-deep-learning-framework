#2 module - binary unary
class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradientwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
