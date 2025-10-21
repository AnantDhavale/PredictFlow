import random

def compute_confidence(step):
    """
    Simulate an NLP-based confidence score between 0 and 1.
    Later: integrate a real model that rates step clarity or success likelihood.
    """
    return round(random.uniform(0.4, 0.95), 2)
