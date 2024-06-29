import re
import numpy as np
from typing import List, Dict, Tuple
 
np.random.seed(42)
class Word2Vec:
    """
    Implements Word2Vec model for word embeddings.
    """

    def __init__(self, text) -> None:
        """
        Initialize the Word2Vec class.
        """
        self.text=text
        
    def _one_hot_encoding(self, idx, vocab_size):
        """
        Generates one-hot encoding for a given index.

        Args:
            idx (int): The index.
            vocab_size (int): The size of the vocabulary.

        Returns:
            List[int]: The one-hot encoding.
        """
        res = [0] * vocab_size
        res[idx] = 1
        return res

    def _concat(self, *args):
        """
        Concatenates multiple iterables.

        Args:
            *args: The iterables to concatenate.

        Yields:
            The concatenated iterables.
        """
        for arg in args:
            yield from arg

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text.

        Args:
            text (str): The input text.

        Returns:
            List[str]: The tokenized words.
        """
        re_text = re.sub(r'[^\w\s]', '', text)
        words = re_text.split()
        words = [word.lower() for word in words]
        return words

    def mapping(self, tokens: List[str]) -> Dict[str, int]:
        """
        Maps tokens to indices.

        Args:
            tokens (List[str]): The tokens.

        Returns:
            Dict[str, int]: The token-to-index mapping.
        """
        word_to_idx = {}
        idx_to_word = {}

        for i, token in enumerate(set(tokens)):
            word_to_idx[token] = i
            idx_to_word[i] = token

        return word_to_idx

    def _generate_training_data(self, tokens: List[str], word_to_idx: Dict[str, int], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates training data.

        Args:
            tokens (List[str]): The tokens.
            word_to_idx (Dict[str, int]): The token-to-index mapping.
            window_size (int): The window size.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The training data.
        """
        X = []
        y = []

        for i in range(len(tokens)):
            idx = self._concat(range(max(0, i-window_size), i), range(i, min(len(tokens), i+window_size+1)))
            for j in idx:
                if i == j:
                    continue
                X.append(self._one_hot_encoding(word_to_idx[tokens[i]], len(word_to_idx)))
                y.append(self._one_hot_encoding(word_to_idx[tokens[j]], len(word_to_idx)))

        return np.array(X), np.array(y)

    def init_weights(self, vocab_size: int, n_embeddings: int) -> Dict[str, np.ndarray]:
        """
        Initializes the weights for the model.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_embeddings (int): The number of embeddings.

        Returns:
            Dict[str, np.ndarray]: The initialized weights.
        """
        model = {
            "w1": np.random.randn(vocab_size, n_embeddings),
            "w2": np.random.randn(n_embeddings, vocab_size)
        }

        return model

    def _softmax(self, X) -> List[np.ndarray]:
        """
        Computes the softmax of the input.

        Args:
            X (np.ndarray): The input.

        Returns:
            List[np.ndarray]: The softmax output.
        """
        res = []
        for x in X:
            exp = np.exp(x)
            res.append(exp / np.sum(exp))
        return res

    def forward(self, model, X, cache=True):
        """
        Performs the forward pass of the model.

        Args:
            model (Dict[str, np.ndarray]): The model weights.
            X (np.ndarray): The input.
            cache (bool, optional): Whether to cache the intermediate results. Defaults to True.

        Returns:
            np.ndarray or Dict[str, np.ndarray]: The output or the cache.
        """
        cache = {}
        
        cache["a1"] = np.dot(X, model["w1"])
        cache["a2"] = np.dot(cache["a1"], model["w2"])
        cache["z"] = self._softmax(cache["a2"])

        if not cache:
            return cache["z"]
        return cache

    def cross_entropy(self, z, y):
        """
        Computes the cross-entropy loss.

        Args:
            z (np.ndarray): The predicted probabilities.
            y (np.ndarray): The true labels.

        Returns:
            float: The cross-entropy loss.
        """
        return -1 * np.sum(y * np.log(z))
    