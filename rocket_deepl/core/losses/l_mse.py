from rocket_deepl.module import Module

class loss(Module):
    def __call__(self, estimated, target):
        return self.forward(estimated, target)

    #TODO:
    def forward(self, estimated, target):
        self.estimated = estimated
        self.target = target
        return ((estimated.data - target.data) ** 2).mean()

    #TODO:
    def backward(self, *noparam):
        return 2 * (self.estimated - self.target)