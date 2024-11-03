from typing import Optional

from tqdm import tqdm # type: ignore
import autograd.numpy as np

from .gradient_descent import GD


class Plain(GD):
    def __init__(
            self, 
            lr: float = 0.01,
            momentum: Optional[float] = 0.0,
            tuner: Optional[str] = None,
            max_iter: int = 50,
        ) -> None:
        super().__init__(lr, momentum, tuner)
        self.max_iter = max_iter

    def gradient_descent(self, input, params, target):
        if self.tuner is not None:
            self.m = [0.0] * len(params)
            self.s = [0.0] * len(params)
        if self.momentum:
            self.delta = [0.0] * len(params)

        i = 0
        grad_mag = np.inf
        with tqdm(total=self.max_iter) as pbar:
            while (i < self.max_iter) and (grad_mag > self.eps):
                gradient = self.gradient(input, params, target)
                params = self.step(gradient, params, i)
                grad_mag = np.max([np.linalg.norm(grad) for grad in gradient])
                i += 1
                pbar.update()
        return params
