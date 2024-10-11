import numpy as np

from .gradient_descent import GD


class PlainMomentum(GD):
    def __init__(self, eta: float, beta_len: int, delta_momentum: float = 0.3, max_iter: int = 50, tol: float = 10e-8, rng: np.random.default_rng = None) -> None:
        super().__init__()
        self.eta = eta      # something about eigenvalues here???
        self.max_iter = max_iter
        self.beta_len = beta_len
        self.tol = tol
        self.delta_momentum = delta_momentum

    def perform(self) -> np.ndarray:
        """
        Performs the descent iteratively.

        Args:
            Tol (float): when to terminate.

        Returns:
            (np.ndarray): beta.
        """
        cost = 10
        # if self.gradient is None:
        #     self.calculate_gradient()
        beta = np.random.randn(self.beta_len)
        change = 0.0
        i = 0
        while (cost > self.tol) and (i < self.max_iter):
            new_change = self.eta*self.gradient(beta) + self.delta_momentum*change
            beta -= new_change
            change = new_change
            i += 1
        return beta