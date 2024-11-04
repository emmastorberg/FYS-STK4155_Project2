from abc import ABC, abstractmethod

import autograd.numpy as np


class GD(ABC):
    def __init__(
            self,
            lr,
            momentum,
            tuner,
        ):
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

    def set_gradient(self, gradient):
        self.gradient = gradient

    def add_momentum(self):
        momentum = self.momentum * self.prev_steps
        return momentum
    
    def adagrad(self, gradient: np.ndarray, i_param: int):
        i = i_param
        self.s[i] += gradient**2
        steps = self.lr*gradient/(np.sqrt(self.s[i]) + self.eps) # jnp.sqrt()
        return steps

    def rmsprop(self, gradient, i_param):
        i = i_param
        rho = 0.99
        self.s[i] = (rho*self.s[i] + (1 - rho)*(gradient**2))
        steps = (self.lr*gradient)/(np.sqrt(self.s[i]) + self.eps)
        return steps
    
    def adam(self, gradient, iteration, i_param):
        iteration = iteration+1
        i = i_param
        beta1, beta2 = 0.9, 0.99
        self.m[i] = beta1*self.m[i] + (1 - beta1)*gradient
        self.s[i] = beta2*self.s[i] + (1 - beta2)*(gradient**2)
        m_hat = self.m[i]/(1 - beta1**iteration)
        s_hat = self.s[i]/(1 - beta2**iteration)
        steps = (self.lr*m_hat)/(np.sqrt(s_hat) + self.eps)
        return steps

    def tune_learning_rate(self, gradient, iteration, i_param) -> np.ndarray:
        if self.tuner == "adagrad":
            steps = self.adagrad(gradient, i_param)

        elif self.tuner == "rmsprop":
            steps = self.rmsprop(gradient, i_param)

        elif self.tuner == "adam":
            steps = self.adam(gradient, iteration, i_param)
        return steps

    def step(self, gradient, params, iteration):
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
        raise NotImplementedError
