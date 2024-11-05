from typing import Optional
import copy

from tqdm import tqdm
import autograd.numpy as np

from .gradient_descent import GD


class Plain(GD):
    """Plain Gradient Descent."""
    def __init__(
            self, 
            lr: float = 0.01,
            momentum: Optional[float] = 0.0,
            tuner: Optional[str] = None,
            max_iter: int = 1000,
            save_info_per_iter: bool = False,
        ) -> None:
        """Initialize the class.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (Optional[float], optional): Momentum parameter. Defaults to 0.0.
            tuner (Optional[str], optional): 
                Optional adaptive learning rate method. Options are "adam", "adagrad" and "rmsprop". Defaults to None.
            max_iter (int, optional): Maximum number of iterations before termination. Defaults to 1000.
            save_info_per_iter (bool, optional): Save the parameters at each iteration. Defaults to False.
        """
        super().__init__(lr, momentum, tuner)
        self.max_iter = max_iter
        self.save_info_per_iter = save_info_per_iter

        self.info = None

    def gradient_descent(self, input: np.ndarray, params: list[np.ndarray], target: np.ndarray) -> list[np.ndarray]:
        """Perform the gradient descent.

        Args:
            input (np.ndarray): Input values.
            params (list[np.ndarray]): Parameters to be optimized.
            target (np.ndarray): Target values.

        Returns:
            list[np.ndarray]: Optimized parameters.
        """
        if self.tuner is not None:
            self.m = [0.0] * len(params)
            self.s = [0.0] * len(params)
        if self.momentum:
            self.delta = [0.0] * len(params)
        if self.save_info_per_iter:
            self.info = [0] * self.max_iter

        i = 0
        low_grad_counter = 0
        grad_mag = np.inf
        with tqdm(total=self.max_iter) as pbar:
            while (i < self.max_iter) and (low_grad_counter < 10):
                gradient = self.gradient(input, params, target)
                params = self.step(gradient, params, i)

                grad_mag = np.max([np.linalg.norm(grad) for grad in gradient])

                if grad_mag < self.eps:
                    low_grad_counter += 1
                else:
                    low_grad_counter = 0

                if self.save_info_per_iter:
                    self.info[i] = copy.deepcopy(params)
                    if low_grad_counter == 10:
                        self.info[i+1:] = [np.copy(params[0])] * (self.max_iter - i)
                i += 1
                pbar.update()

        return params
