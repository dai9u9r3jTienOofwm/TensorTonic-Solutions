import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X)

    if X.ndim < 2 or X.shape[0] < 2:
        return None
    
    n = X.shape[0]
    mean = np.mean(X,axis = 0)
    X_centered = X - mean

    
    return 1 / (n - 1) * X_centered.T @ X_centered