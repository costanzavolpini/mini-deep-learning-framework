class Module(object):
    """
    Abstract class module.
    """

    def forward(self, *input):
        """
        Forward pass
        Input:
            *input: 0 or 1 value.
        """
        raise NotImplementedError

    def backward(self, *gradientwrtoutput):
        """
        Backward pass
        Input:
            *gradientwrtoutput = 0 or 1 value.
        """
        raise NotImplementedError

    def param(self):
        """
        Return parameters
        Output:
            array of parameters
        """
        return []


