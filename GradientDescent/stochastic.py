from typing import Optional

import numpy as np
from tqdm import tqdm

from .gradient_descent import GD


class Stochastic(GD):
    def __init__(
            self, 
            lr: float = 0.01,
            lr_schedule: str = "fixed",
            momentum: Optional[float] = 0.0,
            tuner: Optional[str] = None,
            M: int = 5, 
            n_epochs: int = 50, 
            t: int = 50,
            decay_iter: int = 500,
        ) -> None:
        super().__init__(lr, momentum, tuner)
        self.lr_schedule = lr_schedule

        lr = 0.0
        self.lr0 = lr
        self.M = M
        self.n_epochs = n_epochs
        self.t = t
        self.decay_iter = decay_iter

        if not (lr_schedule in ["fixed", "linear", "minibatch"]):
            raise ValueError

    def learning_schedule(self, minibatch, epoch) -> float:
        if self.lr_schedule == "fixed":
            return self.lr0
        elif self.lr_schedule == "linear":
            kappa = epoch / self.decay_iter
            return (1 - kappa) * self.lr0 + kappa * self.lr0 * 0.01
        # elif self.lr_schedule == "minibatch":
        #     return self.t/(t+self.lr0)

    def gradient_descent(self, input, params, target):
        if self.tuner is not None:
            self.m = [0.0] * len(params)
            self.s = [0.0] * len(params)
        if self.momentum:
            self.delta = [0.0] * len(params)
        m = len(input) // self.M
        for epoch in tqdm(range(self.n_epochs)):
            m_range = np.arange(0, m)
            np.random.shuffle(m_range)

            for i in m_range:
                xi = input[i:i+self.M]
                yi = target[i:i+self.M]
                self.lr = self.learning_schedule(epoch*m + i, epoch)
                gradient = self.gradient(xi, params, yi)
                params = self.step(gradient, params, epoch)

        return params
        