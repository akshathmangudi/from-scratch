import numpy as np

class SelfAttention:
    """
    This class implements the self-attention mechanism for a single head.
    """
    def __init__(self, input_matrix: np.ndarray, seq_length: int = 6, d_model: int = 512):
        self.seq_length = seq_length
        self.d_model = d_model
        self.input_matrix = input_matrix
        self.w_q, self.w_k, self.w_v = self.generate_weights()

    def generate_weights(self): 
        w_q = np.random.rand(self.d_model, self.d_model)
        w_k = np.random.rand(self.d_model, self.d_model)
        w_v = np.random.rand(self.d_model, self.d_model)
        
        return w_q, w_k, w_v

    def forward(self):
        query = np.dot(self.input_matrix, self.w_q)
        key = np.dot(self.input_matrix, self.w_k)
        value = np.dot(self.input_matrix, self.w_v)

        sf = self.d_model**2
        num = np.dot(query, key.T) 
        denom = (1 / np.sqrt(self.d_model)) * num
        output = (1/sf) * denom

        prob_num = np.exp(output - np.max(output, axis=-1, keepdims=True))
        prob_denom = prob_num.sum(axis=-1, keepdims=True)

        final = (prob_num / prob_denom)
        out = sf * np.dot(final, value)

        return out
