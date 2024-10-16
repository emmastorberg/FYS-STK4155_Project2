import numpy as np

from .gradient_descent import GD


class Stochastic(GD):
    def __init__(
            self, 
            eta: float = 0.01,
            eta_tuner: str | None = None,
            delta_momentum: float | None = None,
            max_iter: int = 50,
            tol: float = 10e-8, 
            rng: np.random.Generator | None = None, 
            M: int = 5, 
            num_epochs: int = 50, 
            t0: int = 5, 
            t1: int = 50
        ) -> None:
        super().__init__(eta, eta_tuner, delta_momentum, max_iter, tol, rng)

        self.M = M
        self.num_epochs = num_epochs
        self.t0 = t0
        self.t1 = t1

    def set_gradient(self, X: np.ndarray, y: np.ndarray,  lmbda: float | int = 0) -> None:
        super().set_gradient(X, y, lmbda)
        self.gradient = lambda beta, xi, yi: (2.0/self.M) * xi.T @ (xi @ beta - yi) #changed to M in stochastic
        # legg inn en unpack her?
        if lmbda:
            self.gradient = lambda beta, xi, yi: self.gradient(beta, xi, yi) + 2*lmbda*beta

    def learning_schedule(self, t: int) -> float:
        return self.t0/(t+self.t1)
    
    def perform(self) -> np.ndarray:
        m = int(self.X_num_rows/self.M) # Lag sjekk for at disse er riktige
        beta = self.rng.random(self.X_num_cols)
        delta_0 = 0.0
        iter = 0
        for epoch in range(self.num_epochs):
            m_range = np.arange(0, m - 1)
            self.rng.shuffle(m_range)
            iter += 1
            for k in m_range:
                xk = self.X[k:k+self.M]
                yk = self.y[k:k+self.M]
                eta = self.learning_schedule(epoch*m + k)
                if not self.tune:
                    delta = eta*self.gradient(beta, xk, yk)
                if self.tune:
                    gradient = self.gradient(beta, xk, yk)
                    if self.eta_tuner == "adam":
                        print("iter is", iter)
                        self.t = iter
                    delta = self.tune_learning_rate(gradient)
                if self.momentum:
                    delta, delta_0 = self.add_momentum(delta, delta_0)
                beta -= delta
        return beta


    




        