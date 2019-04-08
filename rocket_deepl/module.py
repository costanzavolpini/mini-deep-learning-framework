#2 module - binary unary
class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradientwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

# class BinaryOperation(Module):
#     def __init__(self):
#         self.l = None
#         self.r = None

#     def __str__(self):
#         return "{} left operator: {}\n right operator:\n {}".format(self.__class__.__name__, self.l, self.r)

#     def __repr__(self):
#         return self.__str__()

#     def __call__(self, l, r):
#         return self.forward(l, r)

#     def forward(self, l, r):
#         self.l = l
#         self.r = r

# class UnaryOperation(Module):
#     def __init__(self):
#         self.input = None

#     def __str__(self):
#         return "{} unary operator: {}\n".format(self.__class__.__name__, self.input)

#     def __repr__(self):
#         return self.__str__()

#     def __call__(self, input):
#         return self.forward(input)

#     def forward(self, input):
#         self.input = input
