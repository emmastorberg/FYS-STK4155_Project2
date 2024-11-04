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
            n_epochs: int = 10,
            decay_iter: int = 5,
            save_info_per_iter: bool = False,
        ) -> None:
        super().__init__(lr, momentum, tuner)
        self.lr_schedule = lr_schedule

        self.lr0 = lr
        self.M = M
        self.n_epochs = n_epochs
        self.decay_iter = decay_iter
        self.save_info_per_iter = save_info_per_iter

        if not (lr_schedule in ["fixed", "linear"]):
            raise ValueError

    def learning_schedule(self, epoch) -> float:
        if self.lr_schedule == "fixed":
            return self.lr0
        elif self.lr_schedule == "linear":
            if epoch < self.decay_iter:
                kappa = epoch / self.decay_iter
                return (1 - kappa) * self.lr0 + kappa * self.lr0 * 0.01
            else:
                return self.lr0 * 0.01

    def gradient_descent(self, input, params, target):
        if self.tuner is not None:
            self.m = [0.0] * len(params)
            self.s = [0.0] * len(params)
        if self.momentum:
            self.delta = [0.0] * len(params)
        if self.save_info_per_iter:
            self.info = [0] * self.n_epochs

        m = len(input) // self.M
        for epoch in tqdm(range(self.n_epochs)):
            m_range = np.arange(0, m)
            np.random.shuffle(m_range)

            for i in m_range:
                xi = input[i:i+self.M]
                yi = target[i:i+self.M]
                self.lr = self.learning_schedule(epoch)
                gradient = self.gradient(xi, params, yi)
                params = self.step(gradient, params, epoch)

            if self.save_info_per_iter:
                # print(params)
                self.info[epoch] = np.copy(params[0])

        return params
        