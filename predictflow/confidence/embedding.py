import numpy as np

def get_vector(text: str):
    """
    Simple placeholder for embeddings.
    Later: replace with sentence-transformers or OpenAI embeddings.
    """
    np.random.seed(abs(hash(text)) % (10 ** 6))  # deterministic for text
    return np.random.rand(128).tolist()
