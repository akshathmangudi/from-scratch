import numpy
from .tensor import Tensor

"""
    These set of functions are defined under a Functions class in 
    nn.py also, but these two implementations can be analogous to 
    torch.nn and torch.nn.functional
"""


class LeakyReLU: 
    def __init__(self, alpha: float = 0.01): 
        self.alpha = alpha

    def forward(self, x): 
        return numpy.maximum(self.alpha*x, x)

class ReLU: 
    def __init__(self): 
        pass

    def forward(self, x): 
        self.x = x
        self.out = numpy.maximum(0, x)
        return self.out

class Tanh: 
    def __init__(self): 
        pass

    def forward(self, x): 
        self.x = x 
        self.out = numpy.tanh(x)

        return self.out

class Sigmoid: 
    def __init__(self): 
        pass

    def forward(self, x): 
        self.x = x
        self.out = 1 / (1 + numpy.exp(-x))

        return self.out

