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
        m = len(input) // self.M
        for epoch in range(self.n_epochs):
            m_range = np.arange(0, m)
            np.random.shuffle(m_range)

            for i in m_range:
                xi = input[i:i+self.M]
                yi = target[i:i+self.M]
                self.lr = self.learning_schedule(epoch*m + i)
                gradient = self.gradient(xi, params, yi)
                params = self.step(gradient, params, epoch)

        return params
        