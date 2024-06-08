import numpy 

"""
    This file contains the implementation of weight initialization techniques. Mainly we will be taking a look at Kaiming, Glorot and Uniform distribution of weight initialization
"""

def uniform(shape: tuple): 
    min_val: float = 0.0
    max_val: float = 1.0

    return numpy.random.uniform(min_val, max_val, shape)


def kaiming(shape: tuple):
    limit = numpy.sqrt(2 / shape[0])
    return numpy.random.uniform(-limit, limit, shape)


def glorot(shape: tuple): 
    limit = numpy.sqrt(6 / (shape[0] + shape[1]))
    return numpy.random.uniform(-limit, limit, shape)
