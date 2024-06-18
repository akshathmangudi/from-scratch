import numpy as np
from . import weights

class SelfAttention:
    def __init__(self, seq_length: int = 6, d_model: int = 512):
        self.seq_length = seq_length
        self.d_model = d_model
        
        self.shape: tuple = (self.seq_length, self.d_model)
        
        self.matrix = weights.uniform(self.shape)
        
        # Defining our Q, K, V matrices.
        self.query = self.matrix
        self.key = self.matrix
        self.value = self.matrix
        
    def operation(self):
        # print(f"This is the query matrix: {self.query}")
        # print(f"This is the key matrix: {self.key}")
        # print(f"This is the value matrix: {self.value}")
        
        self.key = self.key.transpose()
        num = np.dot(self.query, self.key)  # Shape (6, 6)
        denom = (1 / np.sqrt(self.d_model)) * num  # Scaling factor for stability
        
        output = np.dot(denom, self.value)  # Shape (6, 512)
        # print(f"This is the output matrix: {output}")
        
        # Now, to compute the softmax
        prob_num = np.exp(output - np.max(output, axis=-1, keepdims=True))
        prob_denom = prob_num.sum(axis=-1, keepdims=True)
        
        final = prob_num / prob_denom
        print(f"This is the final softmax output: {final}")
        
        return final