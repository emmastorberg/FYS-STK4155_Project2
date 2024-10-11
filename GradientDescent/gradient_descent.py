
import numpy as np

class GD:
    def __init__(self):
        self.gradient = None
        self.beta_len = None
        
    def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
        """
        Setter method??
        """
        n = len(y)
        self.beta_len = len(X[0])
        self.gradient = lambda beta: (2.0/n)*X.T @ (X @ beta - y) + 2*lmbda*beta

    def calculate_gradient(self):
        ...

    def perform(self):
        raise NotImplementedError
