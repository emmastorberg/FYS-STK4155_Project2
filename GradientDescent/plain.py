import numpy as np

from .gradient_descent import GD


class Plain(GD):
    def __init__(
            self, 
            eta: float = 0.01,
            eta_tuner: str | None = None,
            delta_momentum: float | None = None,
            max_iter: int = 50, 
            tol: float = 1e-8, 
            rng: np.random.Generator | None = None,
        ) -> None:
        super().__init__(eta, eta_tuner, delta_momentum, max_iter, tol, rng)

    def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
        super().set_gradient(X, y, lmbda)
        self.gradient = lambda beta: (2.0/self.X_num_rows) * X.T @ (X @ beta - y)
        if lmbda:
            self.gradient = lambda beta: self.gradient(beta) + 2*lmbda*beta

    def perform(self) -> np.ndarray:
        """
        Performs the descent iteratively.

        Args:
            Tol (float): when to terminate.

        Returns:
            (np.ndarray): beta.
        """
        cost = 10
        beta = self.rng.random(self.X_num_cols)
        i = 0
        step_0 = 0.0

        while (cost > self.tol) and (i < self.max_iter):
            i += 1
            if not self.tune:
                step = self.eta * self.gradient(beta)
            if self.tune:
                if self.eta_tuner == "adam":
                    self.t = i 
                gradient = self.gradient(beta)
                step = self.tune_learning_rate(gradient)
            if self.momentum:
                step, step_0 = self.add_momentum(step, step_0)
            beta -= step
            
        return beta

# TODO: changed beta to global self variable, but we may need to change the different gradient calls to implement stochastic