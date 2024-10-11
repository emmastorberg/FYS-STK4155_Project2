import numpy as np
import jax
import jax.numpy as jnp

from jax import grad as jax_grad
from .gradient_descent import GD

class StochasticMomentum(GD):
    def __init__(self, eta: float, beta_len: int, max_iter: int = 50, tol: float = 10e-8, 
                 rng: np.random.default_rng = None, M: int = 5, n_epochs:int = 50, t0: int = 5, t1: int = 50, delta_momentum: float = 0.3, solver: str = "analytical"): 
        super().__init__()
        self.eta = eta      # something about eigenvalues here???
        self.max_iter = max_iter
        self.beta_len = beta_len
        self.tol = tol
        self.rng = rng
        self.X = None
        self.y = None
        self.n = None
        self.init_beta = None
        self.M = M #size of each minibatch
        self.n_epochs = n_epochs
        self.t0 = t0
        self.t1 = t1
        self.solver = solver
        self.delta_momentum = delta_momentum

    def CostOLS(self, y, X, beta):
        return (1.0/self.n)*np.sum((y-X @ beta)**2)

    def set_gradient(self, X: np.ndarray, y: np.ndarray, lmbda: float | int = 0) -> None:
        self.n = len(y)
        self.beta_len = len(X[0])
        self.X = X
        self.y = y
        #if self.solver == "analytical":
        self.gradient = lambda beta, xi, yi: (2.0/self.n)*xi.T @ (xi @ beta - yi) + 2*lmbda*beta
        
        
        """
        elif self.solver == "adam_jax":
            self.gradient = jax_grad(self.CostOLS, 2) #INSERT Function
            self.eta = jnp.float64(self.eta)

        elif self.solver != "jax":
            self.gradient = jax_grad(self.CostOLS) #INSERT Function
            self.eta = jnp.float64(self.eta)
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

