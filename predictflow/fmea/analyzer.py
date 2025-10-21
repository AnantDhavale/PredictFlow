import random

def compute_rpn(step):
    """
    Dummy Failure Mode & Effects Analysis
    RPN = Severity * Occurrence * Detection
    """
    s = random.randint(1, 10)
    o = random.randint(1, 10)
    d = random.randint(1, 10)
    rpn = s * o * d
    return rpn
