import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n = X.shape[0]
    w , b = np.zeros(X.shape[1]),0
    for i in range(steps):
        p = _sigmoid(X @ w + b)

        assert p.shape == y.shape
        
        w_hat = w - lr  * (1/n) * X.T @ (p - y)
        b_hat = b - lr * (1/n) * np.sum(p - y)

        if (np.all(w_hat - w) < 1e-6 and np.abs(b_hat - b) < 1e-6 ):
            break

        w = w_hat
        b = b_hat

    return (w,b)