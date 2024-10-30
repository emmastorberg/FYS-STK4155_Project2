from typing import Optional, Callable

import numpy as np

from .gradient_descent import GD


class Plain(GD):
    def __init__(
            self, 
            lr: float = 0.01,
            momentum: Optional[float] = 0.0,
            tuner: Optional[str] = None,
            max_iter: int = 50,
        ) -> None:
        super().__init__(lr, momentum, tuner)
        self.max_iter = max_iter

    def gradient_descent(self, input, params, target):
        i = 0
        # grad_mag = np.inf

        while (i < self.max_iter): # and (grad_mag > self.eps):
            # print(f"i: {i}")
            # print(f"beta: {params}")
            gradient = self.gradient(input, params, target)
            # print(f"gradient: {gradient} \n")
            # print(f"computed gradient: {gradient}")
            params = self.step(gradient, params)

            # grad_mag = np.linalg.norm(np.sum(gradient), ord=2)
            i += 1
        return params


    # def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
    #     super().set_gradient(X, y, lmbda)
    #     self.gradient = lambda beta: (2.0/self.X_num_rows) * X.T @ (X @ beta - y)
    #     if lmbda:
    #         self.gradient = lambda beta: self.gradient(beta) + 2*lmbda*beta

    # def perform(self, *args) -> np.ndarray:
    #     """
    #     Performs the descent iteratively.

    #     Args:
    #         Tol (float): when to terminate.

    #     Returns:
    #         (np.ndarray): beta.
    #     """
        
    #     termination_condition = ...
    #     beta = self.rng.random(self.X_num_cols)
    #     i = 0
    #     step_0 = 0.0

    #     while (termination_condition) and (i < self.max_iter):
    #         i += 1
    #         gradient = self.gradient(args)
    #         args = self.update(args, gradient)
            
    #     return beta

# TODO: changed beta to global self variable, but we may need to change the different gradient calls to implement stochastic