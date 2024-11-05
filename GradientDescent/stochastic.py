from typing import Optional
import copy

import numpy as np
from tqdm import tqdm

from .gradient_descent import GD


class Stochastic(GD):
    """Stochastic Gradient Descent."""
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
        """Initialize the class.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            lr_schedule (str): Learning rate schedule. Possible values are "fixed" and "linear". Defaults to "fixed".
            momentum (Optional[float], optional): Momentum parameter. Defaults to 0.0.
            tuner (Optional[str], optional): 
                Optional adaptive learning rate method. Options are "adam", "adagrad" and "rmsprop". Defaults to None.
            M (int): Size of mini batces. Defaults to 5.
            n_epochs (int): Number of epochs. Defaults to 10.
            decay_iter (int): If `lr_schedule` is "linear", at which iteration to stop the linear decay. Defaluts to 5.
            save_info_per_iter (bool, optional): Save the parameters at each iteration. Defaults to False.
        """
        super().__init__(lr, momentum, tuner)
        self.lr_schedule = lr_schedule

        self.lr0 = lr
        self.M = M
        self.n_epochs = n_epochs
        self.decay_iter = decay_iter
        self.save_info_per_iter = save_info_per_iter

        if not (lr_schedule in ["fixed", "linear"]):
            raise ValueError

    def learning_schedule(self, epoch: int) -> float:
        """Return learning rate for specified learning rate scedule.

        Args:
            epoch (int): The current epoch.

        Returns:
            float: Learning rate.
        """
        if self.lr_schedule == "fixed":
            return self.lr0
        elif self.lr_schedule == "linear":
            if epoch < self.decay_iter:
                kappa = epoch / self.decay_iter
                return (1 - kappa) * self.lr0 + kappa * self.lr0 * 0.01
            else:
                return self.lr0 * 0.01

    def gradient_descent(self, input, params, target):
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
            self.info = [0] * self.n_epochs

        m = len(input) // self.M
        a = [np.array([1, 2, 3])]
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
                self.info[epoch] = copy.deepcopy(params)

        return params
        