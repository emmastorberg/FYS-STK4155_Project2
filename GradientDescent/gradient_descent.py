from abc import ABC, abstractmethod

import numpy as np

class GD(ABC):
    def __init__(
            self,
            eta: float,
            eta_tuner: str | None,
            delta_momentum: float | None,
            max_iter: int, 
            tol: float, 
            rng: np.random.Generator | None,
        ) -> None:
        """
        Initialize base class for gradient descent methods.

        Args:
            max_iter (int): maximum number of iterations before termination.
            tol (int): terminate when cost is below `tol`.
            rng (np.random.Generator or None): random generator. If None, rng.random.default_rng(None) is used.

        Returns:
            None
        """
        self.eta = eta
        self.eta_tuner = eta_tuner
        self.delta_momentum = delta_momentum
        self.max_iter = max_iter
        self.tol = tol
        if rng is None:
            rng = np.random.default_rng(None)
        self.rng = rng

        if eta_tuner is None:
            self.tune = False
        else:
            if not (eta_tuner in ["adagrad, rmsprop, adam"]):
                raise ValueError
            self.tune = True

        if delta_momentum is None:
            self.momentum = False
        else:
            self.momentum = True

        self.gradient = None
        self.cost = None
        self.X = None
        self.y = None
        self.lmbda = 0
        self.X_num_rows = None
        self.X_num_cols = None
    
    def set_cost(self):
        self.cost = lambda X, y, beta: (1/self.X_num_rows) * np.sum((y - (X @ beta))**2)
        if self.lmbda:
            self.cost = lambda X, y, beta: self.cost(X, y, beta) + self.lmbda * np.sum(beta**2)

    @abstractmethod
    def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
        """
        Setter method??
        """
        self.X = X
        self.y = y
        self.lmbda = lmbda
        self.X_num_rows, self.X_num_cols = X.shape
        self.set_cost()
        # raise NotImplementedError

    @abstractmethod
    def perform(self):
        raise NotImplementedError
    
    def add_momentum(self, delta, delta_0):
        delta += self.delta_momentum * delta_0
        delta_0 = delta
        return delta, delta_0

    def tune_learning_rate(self):
        pass