from abc import ABC, abstractmethod
from typing import Tuple

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
            if not (eta_tuner in ["adagrad", "rmsprop", "adam"]):
                raise ValueError
            self.tune = True
            self.small_val = 1e-8
            self.acc_gradient = 0
            
            
            if eta_tuner == "rmsprop":
                 self.rho = 0.99

            if eta_tuner == "adam":
                 self.beta1 = 0.9
                 self.beta2 = 0.999
                 self.t = 1
                 self.first_moment = 0
                 self.second_moment = 0
        
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

    def tune_learning_rate(self, grad_arg: np.ndarray | Tuple[np.ndarray]):
        if self.eta_tuner == "adagrad":
                    self.acc_gradient += self.gradient(grad_arg)**2
                    delta = self.eta*self.gradient(grad_arg)/(np.sqrt(self.acc_gradient) + self.small_val) # jnp.sqrt()

        elif self.eta_tuner == "rmsprop":
                    self.acc_gradient = (self.rho*self.acc_gradient + (1-self.rho)*self.gradient(grad_arg)**2)
                    delta = self.eta*self.gradient(grad_arg)/ (np.sqrt(self.acc_gradient)+self.small_val) #jnp.sqrt

        elif self.eta_tuner == "adam":
                    self.first_moment = self.beta1*self.first_moment + (1-self.beta1)*self.gradient(grad_arg)
                    self.second_moment = self.beta2*self.second_moment+(1-self.beta2)*self.gradient(grad_arg)**2
                    first_term = self.first_moment/(1.0-self.beta1**self.t)   # should plain also be updated like this?
                    second_term = self.second_moment/(1.0-self.beta2**self.t)  #K

                    delta = self.eta*first_term/(np.sqrt(second_term)+self.small_val) #jnp.sqrt

        return delta 
