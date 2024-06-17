import numpy as np


class Adam:
    def __init__(
        self,
        eta: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """
        This class defines the Adam optimizer. 

        Args:
            eta (float): Known as the learning rate, defaults to 0.01.
            beta_1 (float): The decay rate for first moment estimation, defaults to 0.9.
            beta_2 (float): The decay rate for second moment estimation, defaults to 0.999.
            epsilon (float): This is usually so that the algorithm does not get stuck, defaults to 1e-8.
        """
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_dw, self.v_dw = 0, 0  # mean and variance of weights
        self.m_db, self.v_db = 0, 0  # mean and variance of biases

    def update(self, t, w, b, dw, db):
        # For the weights
        self.m_dw = self.beta_1 * self.m_dw + (1 - self.beta_1) * dw
        self.m_db = self.beta_1 * self.v_dw + (1 - self.beta_1) * db

        # For the bias
        self.v_dw = self.beta_2 * self.v_dw + (1 - self.beta_2) * (dw**2)
        self.v_db = self.beta_2 * self.v_db + (1 - self.beta_2) * db

        # Bias correction step
        mw_corr = self.m_dw / (1 - self.beta_1**t)
        mb_corr = self.m_db / (1 - self.beta_1**t)

        vw_corr = self.v_dw / (1 - self.beta_2**t)
        vb_corr = self.v_db / (1 - self.beta_2**t)

        w -= self.eta * (mw_corr / (np.sqrt(vw_corr) + self.epsilon))
        b -= self.eta * (mb_corr / (np.sqrt(vb_corr) + self.epsilon))

        return w, b
