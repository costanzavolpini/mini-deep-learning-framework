from rocket_deepl.module import UnaryOperation, BinaryOperation
import rocket_deepl.tensor as tensor


class Add(BinaryOperation):
    """
    c = a + b
    Gradient in respect of a will be 1 multiplied by the accumulated gradient.
    Gradient in respect of b will be 1 multiplied by the accumulated gradient.
    """
    def forward(self, l, r):
        super().forward(l, r)
        return tensor.Tensor(l.data + r.data, prev_op = self)


    def backward(self, gradient):
        self.l.backward(gradient * 1)
        self.r.backward(gradient * 1)

class Sub(BinaryOperation):
    """
    c = a - b
    Gradient in respect of a will be 1 multiplied by the accumulated gradient.
    Gradient in respect of b will be -1 multiplied by the accumulated gradient.
    """
    def forward(self, l, r):
        super().forward(l, r)
        return tensor.Tensor(l.data - r.data, prev_op = self)


    def backward(self, gradient):
        self.l.backward(gradient * 1)
        self.r.backward(gradient * (-1))
