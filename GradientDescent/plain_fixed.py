import numpy as np

from .gradient_descent import GD


class PlainFixed(GD):
    def __init__(self, eta: float, beta_len: int, max_iter: int = 50, tol: float = 10e-8, rng: np.random.default_rng = None) -> None:
        super().__init__()
        self.eta = eta      # something about eigenvalues here???
        self.max_iter = max_iter
        self.beta_len = beta_len
        self.tol = tol
        self.rng = rng
        self.init_beta = None

    def perform(self, ) -> np.ndarray:
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
        beta = self.rng.random(self.beta_len)
        self.init_beta = beta
        i = 0
        while (cost > self.tol) and (i < self.max_iter):
            beta -= self.eta * self.gradient(beta)
            i += 1
        print(i)
        return beta

