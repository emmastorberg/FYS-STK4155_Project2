# import numpy as np

# from .gradient_descent import GD


# class PlainMomentum(GD):
#     def __init__(
#             self,
#             eta: float = 0.01,
#             delta_momentum: float = 0.3, 
#             max_iter: int = 50, 
#             tol: float = 10e-8, 
#             rng: np.random.Generator | None = None,
#         ) -> None:
#         super().__init__(eta, max_iter, tol, rng)
#         self.delta_momentum = delta_momentum

#     def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
#         super().set_gradient(X, y, lmbda)
#         self.gradient = lambda beta: (2.0/self.X_num_rows) * X.T @ (X @ beta - y)
#         if lmbda:
#             self.gradient = lambda beta: self.gradient(beta) + 2*lmbda*beta

#     def perform(self) -> np.ndarray:
#         """
#         Performs the descent iteratively.

#         Args:
#             Tol (float): when to terminate.

#         Returns:
#             (np.ndarray): beta.
#         """
#         cost = 10
#         beta = self.rng.random(self.X_num_cols)
#         change = 0.0
#         i = 0
#         while (cost > self.tol) and (i < self.max_iter):
#             new_change = self.eta*self.gradient(beta) + self.delta_momentum*change
#             beta -= new_change
#             change = new_change
#             i += 1
#         return beta