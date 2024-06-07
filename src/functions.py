import numpy
from .autograd import Tensor


class LeakyReLU: 
    def __init__(self): 
        pass

    def forward(self, x): 
        pass


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

