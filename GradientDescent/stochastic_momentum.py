import numpy as np

from gradient_descent import GD

class StochasticMomentum(GD):
    def __init__(self, eta: float, beta_len: int, max_iter: int = 50, tol: float = 10e-8, 
                 rng: np.random.default_rng = None, M: int = 5, n_epochs:int = 50, t0: int = 5, t1: int = 50, delta_momentum: float = 0.3):
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
        self.delta_momentum = delta_momentum

    def set_gradient(self, X: np.ndarray, y: np.ndarray,  lmbda: float | int = 0) -> None:
        self.n = len(y)
        self.beta_len = len(X[0])
        self.X = X
        self.y = y
        self.gradient = lambda beta, xi, yi: (2.0/self.n)*xi.T @ (xi @ beta - yi) + 2*lmbda*beta


    def perform(self) -> np.ndarray:
        m = int(self.n/self.M) # number of minibatches 
        beta = np.random.randn(self.beta_len)
        change = 0.0
        for epoch in range(self.n_epochs):
            m_range = np.arange(0, m-1)
            self.rng.shuffle(m_range)
            for k in m_range:
                xk = self.X[k:k+self.M]
                yk = self.y[k:k+self.M]
                eta = self.learning_schedule(epoch*m+k)
                new_change = eta*self.gradient(beta, xk, yk) + self.delta_momentum*change
                beta -= new_change
                change = new_change
        return beta

    def learning_schedule(self, t: int) -> float:
        return self.t0/(t+self.t1)