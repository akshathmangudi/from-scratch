import numpy as np


class RMSprop:
    """
    This class defines the RMSprop optimizer. 
    
    In theory, the difference between RMSprop and Adam is very slight, 
    mostly changing in bias correction and updation.

    All arguments mean the same thing, as defined in adam.py
    """
    def __init__(
        self,
        eta: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_dw, self.v_dw = 0, 0  # mean and variance of weights
        self.m_db, self.v_db = 0, 0  # mean and variance of biases

    def update(self, t, w, b, dw, db):
        # For the weights
        self.m_dw = self.beta_1 * self.m_dw + (1 - self.beta_1) * (dw**2)
        self.m_db = self.beta_1 * self.v_dw + (1 - self.beta_1) * (db**2)

        # For the bias
        self.v_dw = self.beta_2 * self.v_dw + (1 - self.beta_2) * (dw**2)
        self.v_db = self.beta_2 * self.v_db + (1 - self.beta_2) * (db**2)

        # Bias correction step
        mw_corr = self.m_dw / (1 - self.beta_1)
        mb_corr = self.m_db / (1 - self.beta_1)

        vw_corr = self.v_dw / (1 - self.beta_2)
        vb_corr = self.v_db / (1 - self.beta_2)

        w -= self.eta * (mw_corr / (np.sqrt(mb_corr) + self.epsilon))
        b -= self.eta * (vw_corr / (np.sqrt(vb_corr) + self.epsilon))

        return w, b
