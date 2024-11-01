from typing import Optional

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
        i = 0
        # grad_mag = np.inf

        while (i < self.max_iter): # and (grad_mag > self.eps):
            gradient = self.gradient(input, params, target)
            params = self.step(gradient, params, i)
            # grad_mag = np.linalg.norm(np.sum(gradient), ord=2)
            i += 1
        return params
