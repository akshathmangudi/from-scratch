import numpy as np

class RMSprop:
    """
    This class defines the RMSprop optimizer.
    
    Arguments: 
    eta: learning rate, default to 0.01
    beta: decay rate, default to 0.9
    epsilon: for stability, default to 1e-8
    """
    def __init__(
        self,
        eta: float = 0.01,
        beta: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon

        self.v_dw = 0  # variance of weights
        self.v_db = 0  # variance of biases

    def update(self, w, b, dw, db):
        # For the weights
        self.v_dw = self.beta * self.v_dw + (1 - self.beta) * (dw ** 2)
        # For the bias
        self.v_db = self.beta * self.v_db + (1 - self.beta) * (db ** 2)

        w -= self.eta * (dw / (np.sqrt(self.v_dw) + self.epsilon))
        b -= self.eta * (db / (np.sqrt(self.v_db) + self.epsilon))

        return w, b
