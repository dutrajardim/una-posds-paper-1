import numpy as np

def pca(X: np.ndarray) -> tuple:
    m, n = X.shape
    U = np.zeros((n,n))
    S = np.zeros((n,n))
    
    Sigma = (X.T.dot(X))/m
    U, S, V = np.linalg.svd(Sigma, hermitian=True)
    return (U, S, V)

def project_data(X: np.ndarray, U: np.ndarray, K: int) -> np.ndarray:
    return X.dot(U[:,0:K])

def recover_data(Z: np.ndarray, U: np.ndarray, K: int) -> np.ndarray:
    return  Z.dot(U[:, 0:K].T)