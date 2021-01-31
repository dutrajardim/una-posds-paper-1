import numpy as np

def find_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    idx = np.zeros((X.shape[0],1))
    for i in range(0, X.shape[0]):
        min_idx = np.argmin(np.sum((X[i,:] - centroids)**2, axis=1))
        idx[i] = min_idx
    
    return idx

def compute_centroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, X.shape[1]))
    for i in range(0, K):
        filter = (idx == i).reshape(-1)
        size = np.sum(filter)
        centroids[i,:] = np.sum(X[filter, :], axis=0)/size

    return centroids

def k_means(X: np.ndarray, initial_centroids: np.ndarray, max_iters:int=1000, print_cost:bool=False, max_significance:float=None) -> tuple:
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    cost = 0
    significance = 0

    for i in range(0, max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
        n_cost = distortion_cost(X, centroids, idx)
        significance = cost-n_cost
        cost = n_cost
        
        if print_cost:
            print("Iteration: {}, Cost: {}, Significance: {}".format(i, cost, significance))

        if (max_significance and \
            (significance <= max_significance)) or \
            (significance == 0):
            break


    return (centroids, idx, cost)

def init_centroids(X: np.ndarray, K: int) -> np.ndarray:
    rand_idx = np.random.randint(X.shape[0], size=K)
    return X[rand_idx, :]


def distortion_cost(X: np.ndarray, centroids: np.ndarray, idx: np.ndarray) -> float:
    obs_len = X.shape[0]
    cost = .0
    for i in range(obs_len):
        centroid_idx = int(idx[i][0])
        cost += np.sum((X[i, :] - centroids[centroid_idx, :])**2)
    
    return cost/obs_len