from abc import ABC, abstractmethod

import numpy as np

class GD(ABC):
    def __init__(
            self,
            eta: float,
            eta_tuner: str | None,
            delta_momentum: float | None,
            max_iter: int, 
            tol: float, 
            rng: np.random.Generator | None,
        ) -> None:
        """
        Initialize base class for gradient descent methods.

        Args:
            max_iter (int): maximum number of iterations before termination.
            tol (int): terminate when cost is below `tol`.
            rng (np.random.Generator or None): random generator. If None, rng.random.default_rng(None) is used.

        Returns:
            None
        """
        self.eta = eta
        self.eta_tuner = eta_tuner
        self.delta_momentum = delta_momentum
        self.max_iter = max_iter
        self.tol = tol
        if rng is None:
            rng = np.random.default_rng(None)
        self.rng = rng

        if eta_tuner is None:
            self.tune = False
        else:
            if not (eta_tuner in ["adagrad", "rmsprop", "adam"]):
                raise ValueError
            self.tune = True
            self.small_val = 1e-8
            self.acc_gradient = 0
            
            if eta_tuner == "rmsprop":
                 self.rho = 0.99

            if eta_tuner == "adam":
                 self.beta1 = 0.9
                 self.beta2 = 0.999
        
        if delta_momentum is None:
            self.momentum = False
        else:
            self.momentum = True

        self.gradient = None
        self.beta = None
        self.cost = None
        self.X = None
        self.y = None
        self.lmbda = 0
        self.X_num_rows = None
        self.X_num_cols = None

    
    def set_cost(self):
        self.cost = lambda X, y, beta: (1/self.X_num_rows) * np.sum((y - (X @ beta))**2)
        if self.lmbda:
            self.cost = lambda X, y, beta: self.cost(X, y, beta) + self.lmbda * np.sum(beta**2)

    @abstractmethod
    def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
        """
        Setter method??
        """
        self.X = X
        self.y = y
        self.lmbda = lmbda
        self.X_num_rows, self.X_num_cols = X.shape
        self.set_cost()
        # raise NotImplementedError

    @abstractmethod
    def perform(self):
        raise NotImplementedError
    
    def add_momentum(self, delta, delta_0):
        delta += self.delta_momentum * delta_0
        delta_0 = delta
        return delta, delta_0

    def tune_learning_rate(self):
         #acc_gradient = jnp.zeros_like(beta)  # Initialize the gradient accumulator, not sure if shape is correct, Morten uses just a number zero? 
        if self.eta_tuner == "adagrad":
                    #self.beta = jnp.float64(beta) 
                    self.acc_gradient += self.gradient(self.beta)**2
                    delta = self.eta*self.gradient(self.beta)/(np.sqrt(self.acc_gradient) + self.small_val) # jnp.sqrt()
        return delta 

     
"""
def perform(self) -> np.ndarray:
        m = int(self.n/self.M) # number of minibatches 
        beta = np.random.randn(self.beta_len)
        change = 0.0

        # For adaptible learning rate
        small_val = 1e-8
        acc_gradient= 0 #jnp.zeros_like(beta)  # Initialize the gradient accumulator, not sure if shape is correct Morten uses just a number? 

        # For rmsprop
        rho = 0.99

        # For adam
        beta1 = 0.9
        beta2 = 0.999

        for epoch in range(self.n_epochs):
            m_range = np.arange(0, m-1)
            self.rng.shuffle(m_range)

            for k in m_range:
                xk = self.X[k:k+self.M]
                yk = self.y[k:k+self.M]
                eta = self.learning_schedule(epoch*m+k) # or self.eta

                if self.solver == "analytical":
                    new_change = eta*self.gradient(beta, xk, yk) + self.delta_momentum*change

                if self.solver == "ada":
                    self.beta = jnp.float64(beta)
                    acc_gradient += self.gradient(beta, xk, yk)**2
                    new_change = eta*self.gradient(beta, xk, yk)/(jnp.sqrt(acc_gradient) + small_val) + self.delta_momentum*change # Plus moment?
                
                elif self.solver == "rmsprop":
                    # Scaling with rho the new and the previous results
                    acc_gradient = (rho*acc_gradient + (1-rho)*self.gradient**2)
                    new_change = eta*self.gradient/ (jnp.sqrt(acc_gradient)+small_val) #same as for ada but different acc gradient, + Plus moment?

                elif self.solver == "adam":
                    # Computing moments first
                    
                    first_moment = beta1*first_moment + (1-beta1)*self.gradient
                    second_moment = beta2*second_moment+(1-beta2)*self.gradient**2
                    first_term = first_moment/(1.0-beta1**k)
                    second_term = second_moment/(1.0-beta2**k)

                    # Scaling with rho the new and the previous results
                    new_change = eta*first_term/(jnp.sqrt(second_term)+small_val) + self.delta_momentum*change #Plus moment

                beta -= new_change
                change = new_change  #Only for analytical this has purpose
        return beta

    def learning_schedule(self, t: int) -> float:
        return self.t0/(t+self.t1)

"""