import numpy as np

def feature_normalize(X: np.ndarray) -> tuple:
    mu = X.mean(axis=0)
    X_norm = X - mu
    sigma = X_norm.std(axis=0)
    X_norm = X_norm/sigma
    return (X_norm, sigma, mu)