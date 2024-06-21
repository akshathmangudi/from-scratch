import numpy as np

class SelfAttention:
    """
    
    This class defines the self-attention module for a single head. The extension of this class would be the Multi-Head Attention that's used in 
    the transformers architecture. 
    
    The arguments required for this class is simply just the sequence length of the sentence (seq_length) and the embedding dimensions, known as 
    d_model. 
    
    We use the formula: softmax(Q*K.T)*V/sqrt(d_model) to compute the attention matrix. 
    K.T stands for transpose of the key matrix. But it is to be noted that Q, K, V are all the same matrices. 
    """
    def __init__(self, input_matrix: np.ndarray, seq_length: int = 6, d_model: int = 512):
        self.seq_length = seq_length
        self.d_model = d_model
        self.input_matrix = input_matrix
    
    def generate_weights(self): 
        w_q = np.random.rand(self.d_model, self.d_model)
        w_k = np.random.rand(self.d_model, self.d_model)
        w_v = np.random.rand(self.d_model, self.d_model)
        
        return w_q, w_k, w_v
    
    def generate_qkv(self): 
        query = np.dot(self.input_matrix, self.w_q)
        key = np.dot(self.input_matrix, self.w_k)
        value = np.dot(self.input_matrix, self.w_v)
        
        return query, key, value
        
    def operation(self):
        self.w_q, self.w_k, self.w_v = self.generate_weights()
        self.query, self.key, self.value = self.generate_qkv()
        
        # print(f"This is the query matrix: {self.query}")
        # print(f"This is the key matrix: {self.key}")
        # print(f"This is the value matrix: {self.value}")
        
        # As there is a problem with exponentiating large numbers, leading to an overflow 
        # We will scale it down in the middle and scale it back up in the end.
        sf = self.d_model**2
        
        
        self.key_T = self.key.T
        num = np.dot(self.query, self.key_T)  # Shape (6, 6)
        denom = (1 / np.sqrt(self.d_model)) * num  # Scaling factor for stability
        
        output = (1/sf) * denom # Shape (6, 512)
        # print(f"This is the scaled output matrix: {output}")
        
        # For our probability, we will now scale the output 
        
        # Now, to compute the softmax
        prob_num = np.exp(output - np.max(output, axis=-1, keepdims=True))
        prob_denom = prob_num.sum(axis=-1, keepdims=True)
        
        final = (prob_num / prob_denom)
        out = sf * np.dot(final, self.value)
        # print(f"This is the final softmax output: {final}")
        
        return out