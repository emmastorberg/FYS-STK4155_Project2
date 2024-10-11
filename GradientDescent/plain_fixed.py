import numpy as np

import GradientDescent

class PlainFixed(GradientDescent):
    def __init__(self, eta: float, max_iter: int = 50, tol: float = 10e-8) -> None:
        super().__init__()
        self.eta = eta      # something about eigenvalues here???
        self.max_iter = max_iter
        self.tol = tol

    def perform(self, ) -> np.ndarray:
        """
        Performs the descent iteratively.

        Args:
            Tol (float): when to terminate.

        Returns:
            (np.ndarray): beta.
        """
        cost = ...
        # if self.gradient is None:
        #     self.calculate_gradient()
        beta = np.random(np.shape(self.gradient))
        i = 0
        while (cost > self.tol) and (i < self.max_iter):
            beta -= self.eta * self.gradient
            i += 1
        return beta

