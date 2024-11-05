from abc import ABC, abstractmethod
from typing import Callable

import autograd.numpy as np


class GD(ABC):
    """
    Base class for gradient descent.
    """
    def __init__(
            self,
            lr: float,
            momentum: float,
            tuner: str,
        ):
        """Initialize base class.

        Args:
            lr (float): learning rate.
            momentum (flaot): momentum parameter.
            tuner (str): optional adaptive learning rate method. Options are "adam", "adagrad" and "rmsprop".

        Raises:
            ValueError: Raise error if tuner is one of the ones specified above.
        """
        self.lr = lr
        self.momentum = momentum
        self.tuner = tuner

        self.gradient = None
        self.m = None      # first momentum
        self.s = None      # second momentum
        self.eps = 1e-8
        self.delta = None

        if tuner is not None:
            if not (tuner in ["adagrad", "rmsprop", "adam"]):
                raise ValueError

    def set_gradient(self, gradient: Callable):
        """Set gradient for gradient descent.

        Args:
            gradient (Callable): gradient.
        """
        self.gradient = gradient

    def add_momentum(self) -> np.ndarray:
        """Add momentum to step.

        Returns:
            np.ndarray: momentum
        """
        momentum = self.momentum * self.prev_steps
        return momentum
    
    def adagrad(self, gradient: np.ndarray, i_param: int) -> np.ndarray:
        """Tune learning rate with AdaGrad.

        Args:
            gradient (np.ndarray): calculated gradient for current parameter and iteration.
            i_param (int): index of parameter.

        Returns:
            np.ndarray: step size (potential momentum excluded).
        """
        i = i_param
        self.s[i] += gradient**2
        steps = self.lr*gradient/(np.sqrt(self.s[i]) + self.eps)
        return steps

    def rmsprop(self, gradient: np.ndarray, i_param: int) -> np.ndarray:
        """Tune learning rate with RMSProp.

        Args:
            gradient (np.ndarray): calculated gradient for current parameter and iteration.
            i_param (int): index of parameter.

        Returns:
            np.ndarrray: step size (potential momentum excluded).
        """
        i = i_param
        rho = 0.99
        self.s[i] = (rho*self.s[i] + (1 - rho)*(gradient**2))
        steps = (self.lr*gradient)/(np.sqrt(self.s[i]) + self.eps)
        return steps
    
    def adam(self, gradient: np.ndarray, iteration: int, i_param: int) -> np.ndarray:
        """Tune learning rate with Adam.

        Args:
            gradient (np.ndarray): calculated gradient for current parameter and iteration.
            iteration (int): the current epoch/iteration.
            i_param (int): index of parameter.

        Returns:
            np.ndarray: step size (potential momentum excluded).
        """
        iteration = iteration+1
        i = i_param
        beta1, beta2 = 0.9, 0.99
        self.m[i] = beta1*self.m[i] + (1 - beta1)*gradient
        self.s[i] = beta2*self.s[i] + (1 - beta2)*(gradient**2)
        m_hat = self.m[i]/(1 - beta1**iteration)
        s_hat = self.s[i]/(1 - beta2**iteration)
        steps = (self.lr*m_hat)/(np.sqrt(s_hat) + self.eps)
        return steps

    def tune_learning_rate(self, gradient: np.ndarray, iteration: int, i_param: int) -> np.ndarray:
        """Tune the learning rate with the method specified in the constructor.

        Args:
            gradient (np.ndarray): calculated gradient for current parameter and iteration.
            iteration (int): the current epoch/iteration.
            i_param (int): index of parameter.

        Returns:
            np.ndarray: step size (potential momentum excluded).
        """
        if self.tuner == "adagrad":
            steps = self.adagrad(gradient, i_param)

        elif self.tuner == "rmsprop":
            steps = self.rmsprop(gradient, i_param)

        elif self.tuner == "adam":
            steps = self.adam(gradient, iteration, i_param)
        return steps

    def step(self, gradient: np.ndarray, params: list[np.ndarray], iteration: int) -> list[np.ndarray]:
        """Perform a single gradient descent step.

        Args:
            gradient (np.ndarray): calculated gradient for current parameter and iteration.
            params (list[np.ndarray]): The current parameters to optimize.
            iteration (int): the current epoch/iteration.

        Returns:
            list[np.ndarray]: The updated parameters.
        """
        for i, (param, grad) in enumerate(zip(params, gradient)):
            if self.tuner is None:
                steps = self.lr * grad
            else:
                steps = self.tune_learning_rate(grad, iteration, i)
            if self.momentum:
                steps += self.momentum * self.delta[i]
                self.delta[i] = steps
            param -= steps
        return params
    
    @abstractmethod
    def gradient_descent(self):
        """Abstact method for performing the gradient descent.

        Raises:
            NotImplementedError: Raise NotImplementedError if method is not implemented in child class.
        """
        raise NotImplementedError
