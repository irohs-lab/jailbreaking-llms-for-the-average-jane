import numpy as np
from scipy.linalg import cholesky, solve_triangular


class BaseRegressionOracle:

    def __init__(self):
        pass

    def predict(self, x:np.ndarray)->np.float64:
        pass

    def update(self, x:np.ndarray, y:np.float64)->None:
        pass

    def regret_bound(self, **args)->np.float64:
        pass


class VovkOnlineRegressionOracle(BaseRegressionOracle):
    """
        V. Vovk, NeurIPS 1997: Competitive On-Line Linear Regression
    """

    def __init__(self, dim:int, num_rounds:int):
        self.d = dim
        self.b = np.zeros(self.d)
        self.weights_history = []
        self.T = num_rounds
        c = self.T/self.d
        self.a = 0.5*(-c + np.sqrt(c**2 + 4*c))
        self.A = self.a*np.eye(self.d)
    
    def predict(self, x:np.ndarray):

        A = self.A + np.outer(x,x)
        R = cholesky(A, lower=False, check_finite=False) # A = R^T@R

        # weight vector = A_inv b
        v = solve_triangular(R,self.b,trans='T', lower=False, check_finite=False) # forward solve R^T v = b
        w = solve_triangular(R, v, trans='N', lower=False, check_finite=False) # backward solve R w = v

        pred = np.dot(x, w)

        self.weights_history.append(w.copy())

        return pred
    
    def update(self, x:np.ndarray, y:float):

        self.A+=np.outer(x,x)
        self.b += y*x
    
    def regret_bound(self, T):
        return self.d*np.log(T/self.d)


