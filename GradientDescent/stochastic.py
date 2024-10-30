from typing import Optional

import numpy as np

from .gradient_descent import GD


class Stochastic(GD):
    def __init__(
            self, 
            lr: float = 0.01,
            momentum: Optional[float] = 0.0,
            tuner: Optional[str] = None,
            M: int = 5, 
            n_epochs: int = 50, 
            t0: int = 5, 
            t1: int = 50
        ) -> None:
        super().__init__(lr, momentum, tuner)

        self.M = M
        self.n_epochs = n_epochs
        self.t0 = t0
        self.t1 = t1

    def learning_schedule(self, t: int) -> float:
        return self.t0/(t+self.t1)

    def gradient_descent(self, input, params, target):
        print("Descending...")
        m = len(input) // self.M

        for epoch in range(self.n_epochs):
            m_range = np.arange(0, m - 1)
            np.random.shuffle(m_range)

            for i in m_range:
                xi = input[i:i+self.M]
                yi = target[i:i+self.M]
                self.lr = self.learning_schedule(epoch*m + i)
                gradient = self.gradient(xi, params, yi)
                params = self.step(gradient, params)

        print("Done descending!\n")
        return params

                
        

    # def set_gradient(self, X: np.ndarray, y: np.ndarray,  lmbda: float | int = 0) -> None:
    #     super().set_gradient(X, y, lmbda)
    #     self.gradient = lambda beta, xi, yi: (2.0/self.M) * xi.T @ (xi @ beta - yi) #changed to M in stochastic
    #     # legg inn en unpack her?
    #     if lmbda:
    #         self.gradient = lambda beta, xi, yi: self.gradient(beta, xi, yi) + 2*lmbda*beta

  
    
    # def perform(self) -> np.ndarray:
    #     m = int(self.X_num_rows/self.M) # Lag sjekk for at disse er riktige
    #     beta = self.rng.random(self.X_num_cols)
    #     delta_0 = 0.0
    #     iter = 0
    #     for epoch in range(self.num_epochs):
    #         m_range = np.arange(0, m - 1)
    #         self.rng.shuffle(m_range)
    #         iter += 1
    #         for k in m_range:
    #             xk = self.X[k:k+self.M]
    #             yk = self.y[k:k+self.M]
    #             lr = self.learning_schedule(epoch*m + k)
    #             if not self.tune:
    #                 delta = lr*self.gradient(beta, xk, yk)
    #             if self.tune:
    #                 gradient = self.gradient(beta, xk, yk)
    #                 if self.lr_tuner == "adam":
    #                     self.t = iter
    #                 delta = self.tune_learning_rate(gradient)
    #             if self.momentum:
    #                 delta, delta_0 = self.add_momentum(delta, delta_0)
    #             beta -= delta
    #     return beta


    




        