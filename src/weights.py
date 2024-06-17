import numpy 


def uniform(shape: tuple): 
    """ 
    
    This function deals with intializing waits as a uniform distribution.

    Args:
        shape (tuple): takes in a weight matrix

    Returns:
        _type_: returns a matrix of the same shape, with a distribution of weights. 
    """
    min_val: float = 0.0
    max_val: float = 1.0

    return numpy.random.uniform(min_val, max_val, shape)


def kaiming(shape: tuple):
    """
    This function initializes weights under the kaiming/he distribution where shape[0] indicates fan_in

    Arguments are defined exactly in the same context as uniform()
    """
    limit = numpy.sqrt(2 / shape[0])
    return numpy.random.uniform(-limit, limit, shape)


def glorot(shape: tuple): 
    """
    This function initializes weights under the glorot distribution where 
    
    shape[0] = fan_in
    shape[1] = fan_out

    Arguments are defined exacly in the same context as uniform()
    """
    limit = numpy.sqrt(6 / (shape[0] + shape[1]))
    return numpy.random.uniform(-limit, limit, shape)
