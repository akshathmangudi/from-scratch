import numpy as np

class SelfAttention:
    """
    This class implements the self-attention mechanism for a single head.

    Args:
        input_matrix (ndarray): The input matrix of shape (d_model, seq_length).
        seq_length (int, optional): The sequence length. Defaults to 6.
        d_model (int, optional): The dimensionality of the model. Defaults to 512.

    Attributes:
        seq_length (int): The sequence length.
        d_model (int): The dimensionality of the model.
        input_matrix (ndarray): The input matrix of shape (d_model, seq_length).

    Methods:
        forward(): Performs the forward pass of the self-attention mechanism.
            Returns:
                ndarray: The output tensor of shape (d_model, seq_length).
    """
    def __init__(self, input_matrix: np.ndarray, seq_length: int = 6, d_model: int = 512):
        self.seq_length = seq_length
        self.d_model = d_model
        self.input_matrix = input_matrix
        
    def forward(self):
        matrix = np.random.rand(self.d_model, self.seq_length)
        
        query, key, value = matrix, matrix, matrix
        
        sf = self.d_model**2
        num = np.dot(query, key.T) 
        denom = (1 / np.sqrt(self.d_model)) * num
        output = (1/sf) * denom

        prob_num = np.exp(output - np.max(output, axis=-1, keepdims=True))
        prob_denom = prob_num.sum(axis=-1, keepdims=True)

        final = (prob_num / prob_denom)
        out = sf * np.dot(final, value)

        return out
    
    
class MultiHeadAttention:
    """
    This class implements the Multi-Head Attention mechanism.

    Attributes:
        seq_length (int): The sequence length.
        d_model (int): The dimensionality of the model.
        input_matrix (ndarray): The input matrix of shape (d_model, seq_length).
        w_q (ndarray): The weight matrix for query.
        w_k (ndarray): The weight matrix for key.
        w_v (ndarray): The weight matrix for value.

    Methods:
        generate_weights(): Generates the weight matrices for query, key and value.
        forward(): Performs the forward pass of the Multi-Head Attention mechanism.
        
    It is to be noted that usually, w_q, w_k and w_v are learnt parameters, not randomly
    generated matrices. For the sake of simplicity, I have written it like this. 
    """
    def __init__(self, input_matrix: np.ndarray, seq_length: int = 6, d_model: int = 512):
        """
        Initialize the MultiHeadAttention class.

        Args:
            input_matrix (ndarray): The input matrix of shape (d_model, seq_length).
            seq_length (int, optional): The sequence length. Defaults to 6.
            d_model (int, optional): The dimensionality of the model. Defaults to 512.
        """
        self.seq_length = seq_length
        self.d_model = d_model
        self.input_matrix = input_matrix
        self.w_q, self.w_k, self.w_v = self.generate_weights()

    def generate_weights(self):
        """
        Generates the weight matrices for query, key and value.

        Returns:
            ndarray, ndarray, ndarray: The weight matrices for query, key and value.
        """
        w_q = np.random.rand(self.d_model, self.seq_length)
        w_k = np.random.rand(self.d_model, self.seq_length)
        w_v = np.random.rand(self.d_model, self.seq_length)

        return w_q, w_k, w_v

    def forward(self):
        """
        Performs the forward pass of the Multi-Head Attention mechanism.

        Returns:
            ndarray: The output tensor of shape (d_model, seq_length).
        """
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