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
    
    def adagrad(self, gradient: np.ndarray, i: int):
        self.s[i] += gradient**2
        steps = self.lr*gradient/(np.sqrt(self.s[i]) + self.eps) # jnp.sqrt()
        return steps

    def rmsprop(self, gradient, i):
        rho = 0.99
        self.s[i] = (rho*self.s[i] + (1 - rho)*(gradient**2))
        steps = (self.lr*gradient)/(np.sqrt(self.s[i]) + self.eps) #jnp.sqrt
        return steps
    
    def adam(self, gradient, t, i):
        t = t+1
        beta1, beta2 = 0.9, 0.99
        self.m[i] = beta1*self.m[i] + (1 - beta1)*gradient
        self.s[i] = beta2*self.s[i] + (1 - beta2)*(gradient**2)
        m_hat = self.m[i]/(1 - beta1**t)
        s_hat = self.s[i]/(1 - beta2**t)
        steps = (self.lr*m_hat)/(np.sqrt(s_hat) + self.eps) #jnp.sqrt
        return steps

    def tune_learning_rate(self, gradient, iter, i) -> np.ndarray:
        if self.tuner == "adagrad":
            steps = self.adagrad(gradient, i)

        elif self.tuner == "rmsprop":
            steps = self.rmsprop(gradient, i)

        elif self.tuner == "adam":
            steps = self.adam(gradient, iter, i)
        return steps

    def step(self, gradient, params, iter):
        # print(f"gradient: {gradient}")
        for i, (param, grad) in enumerate(zip(params, gradient)):
            if self.tuner is None:
                steps = self.lr * grad
            else:
                steps = self.tune_learning_rate(grad, iter, i)
            if self.momentum:
                steps += self.momentum * self.delta[i]
                self.delta[i] = steps
            param -= steps
        return params
    
    @abstractmethod
    def gradient_descent(self):
        raise NotImplementedError



        






# class GD(ABC):
#     def __init__(
#             self,
#             cost_func,
#             cost_grad: tuple[callable, int | list],
#             lr: float,
#             lr_tuner: str | None,
#             delta_momentum: float | None,
#             max_iter: int,
#             tol: float,
#             rng: np.random.Generator | None,
#         ) -> None:
#         """
#         Initialize base class for gradient descent methods.

#         Args:
#             max_iter (int): maximum number of iterations before termination.
#             tol (int): terminate when cost is below `tol`.
#             rng (np.random.Generator or None): random generator. If None, rng.random.default_rng(None) is used.

#         Returns:
#             None
#         """
#         self.cost_func, self.wrt = cost_func
#         if not isinstance(self.wrt, Iterable):
#             self.wrt = [self.wrt]
#         self.cost_grad = cost_grad
#         self.lr = lr
#         self.lr_tuner = lr_tuner
#         self.delta_momentum = delta_momentum
#         self.max_iter = max_iter
#         self.tol = tol

#         if rng is None:
#             rng = np.random.default_rng(None)
#         self.rng = rng

#         self.params = signature(cost_grad)

#         if lr_tuner is None:
#             self.tune = False
#         else:
#             if not (lr_tuner in ["adagrad", "rmsprop", "adam"]):
#                 raise ValueError
#             self.tune = True
#             self.eps = 1e-8
#             self.s = 0          # second moment of the gradient of the cost
            
#             if lr_tuner == "rmsprop":
#                  self.rho = 0.99

#             if lr_tuner == "adam":
#                  self.beta1 = 0.9
#                  self.beta2 = 0.99
#                  self.t = 0
#                  self.m = 0     # first moment of the gradient of the cost
        
#         if delta_momentum is None:
#             self.momentum = False
#         else:
#             self.momentum = True

#         self.gradient = None
#         self.cost = None
#         self.X = None
#         self.y = None
#         self.lmbda = 0
#         self.X_num_rows = None
#         self.X_num_cols = None


#     def cost(self, predict, target):
#         return self.cost_func(predict, target)
    
#     def gradient(self):
#         return self.cost_grad(args)
    
#     # def set_cost(self):
#     #     self.cost = lambda X, y, beta: (1/self.X_num_rows) * np.sum((y - (X @ beta))**2)
#     #     if self.lmbda:
#     #         self.cost = lambda X, y, beta: self.cost(X, y, beta) + self.lmbda * np.sum(beta**2)

#     @abstractmethod
#     def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
#         """
#         Setter method??
#         """
#         self.X = X
#         self.y = y
#         self.lmbda = lmbda
#         self.X_num_rows, self.X_num_cols = X.shape
#         self.set_cost()
#         # raise NotImplementedError

#     @abstractmethod
#     def perform(self):
#         raise NotImplementedError
    
#     def add_momentum(self, delta, delta_0):
#         delta += self.delta_momentum * delta_0
#         delta_0 = delta
#         return delta, delta_0

#     def tune_learning_rate(self, gradient) -> np.ndarray:
#         if self.lr_tuner == "adagrad":
#             self.s += gradient**2
#             delta = self.lr*gradient/(np.sqrt(self.s) + self.eps) # jnp.sqrt()

#         elif self.lr_tuner == "rmsprop":
#             self.s = (self.rho*self.s + (1 - self.rho)*(gradient**2))
#             delta = (self.lr*gradient)/(np.sqrt(self.s) + self.eps) #jnp.sqrt

#         elif self.lr_tuner == "adam":
#             self.m = self.beta1*self.m + (1 - self.beta1)*gradient
#             self.s = self.beta2*self.s + (1 - self.beta2)*(gradient**2)
#             m_hat = self.m/(1 - self.beta1**self.t)   # should plain also be updated like this?
#             s_hat = self.s/(1 - self.beta2**self.t)  #not sure about value of t, depends on plain or stochastic?
#             delta = (self.lr*m_hat)/(np.sqrt(s_hat) + self.eps) #jnp.sqrt

#         return delta 
    
#     def update(self, args, gradient):
#         if self.tune:
#             step = self.tune_learning_rate(gradient)
#         else:
#             step = self.lr * gradient
#         if self.momentum:
#             step, step_0 = self.add_momentum(step, step_0)
#         args[self.wrt] -= step
#         return param
