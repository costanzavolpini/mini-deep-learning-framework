from rocket_deepl.module import UnaryOperation, BinaryOperation
import rocket_deepl.tensor as tensor

class Add(BinaryOperation):

    def forward(self, l, r):
        super().forward(l, r)
        return tensor.Tensor(l.data + r.data, prev_op = self)


    def backward(self, gradient):
        self.l.backward(gradient * 1)
        self.r.backward(gradient * 1)

class Sub(BinaryOperation):
    def forward(self, l, r):
        super().forward(l, r)
        return tensor.Tensor(l.data - r.data, prev_op = self)


    def backward(self, gradient):
        self.l.backward(gradient * 1)
        self.r.backward(gradient * (-1))


class Div(BinaryOperation):
    def forward(self, l, r):
        super().forward(l, r)
        return tensor.Tensor(l.data /r.data, prev_op = self)


    def backward(self, gradient):
        pass


class Multiply(BinaryOperation):
    def forward(self, l, r):
        super().forward(l, r)
        return tensor.Tensor(l.data *r.data, prev_op = self)


    def backward(self, gradient):
        pass