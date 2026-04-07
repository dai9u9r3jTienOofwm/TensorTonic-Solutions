import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)

    if np.sum(p) != float(1):
        raise ValueError("ValueError")
    
    return np.dot(x,p)
