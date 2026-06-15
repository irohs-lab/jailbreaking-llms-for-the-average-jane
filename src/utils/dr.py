import numpy as np
from sklearn.decomposition import PCA
from DiffRed import DiffRed
from DiffRed.utils import calculate_singular_values
import warnings


def reduce_dimensionality(solver:str, X:np.ndarray, target_dim:int, seed:int)->np.ndarray:
    
    if solver is None: return X

    if solver == 'pca':
        return pca(X,target_dim)
    elif solver == 'rmap':
        return rmap(X,target_dim,seed)
    elif solver == 'diffred':
        return diffred(X,target_dim)
    else:
        raise NotImplementedError(f"Solver {solver} not implemented for dimensionality reduction")


def pca(X:np.ndarray, target_dim:int)->np.ndarray:
    pca = PCA(n_components=target_dim)
    shape = list(X.shape)
    _X = X.reshape(-1, X.shape[-1])
    X_r = pca.fit_transform(_X)
    X_r = X_r.reshape(*(shape[:-1]+[target_dim]))

    return X_r

def rmap(X:np.ndarray, target_dim:int, seed:int)->np.ndarray:
    shape = list(X.shape)
    _X = X.reshape(-1, X.shape[-1])
    n,D = _X.shape[0], _X.shape[1]
    rng = np.random.default_rng(seed = seed)
    G = rng.normal(0,1/np.sqrt(D), (D,target_dim))

    X_r =  np.sqrt(D/target_dim) * _X @ G
    X_r = X_r.reshape(*(shape[:-1]+[target_dim]))

    return X_r

def diffred(X:np.ndarray, target_dim:int):
    shape = list(X.shape)
    _X = X.reshape(-1,X.shape[-1])
    n,D = _X.shape[0], _X.shape[1]
    d = target_dim
    sigma = calculate_singular_values(_X)
    sigma_sq = sigma**2
    bound_vals = []
    # find k1,k2 pair that minimizes stress bound
    for k1 in range(d):
        p = (sigma_sq[:k1].sum())/(sigma_sq.sum())
        p = np.clip(p, 0.0, 1.0)
        k2 = target_dim-k1
        if k2 == 0: continue
        bound_val = np.sqrt(1-p)/np.sqrt(k2)
        bound_vals.append((k1,k2,bound_val))
    
    min_val = min(bound_vals, key=lambda x:x[2])
    k1,k2 = min_val[0], min_val[1]

    dr = DiffRed(k1,k2)
    X_r = dr.fit_transform(_X)

    X_r = X_r.reshape(*(shape[:-1]+[target_dim]))

    return X_r

