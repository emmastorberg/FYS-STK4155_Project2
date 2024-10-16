import pytest # type: ignore

import numpy as np
from sklearn.linear_model import SGDRegressor # type: ignore

from GradientDescent import Plain, Stochastic


def test_simple_test():
    assert 1 + 1 == 2

# def test_plain_fixed_perform() -> None:
#     """
#     Not correct, but close.
#     """
#     n = 100
#     rng = np.random.default_rng(10)
#     x = 2*rng.standard_normal(n)
#     y = 4+3*x+rng.standard_normal(n)
#     X = np.c_[np.ones((n,1)), x]

#     GD = PlainFixed(0.1, beta_len=2, max_iter=200, rng=rng)
#     gradient = lambda beta: (2.0/n)*X.T @ (X @ beta-y)
#     GD.set_gradient(gradient)
#     beta = GD.perform()

#     x = x.reshape(-1, 1)
#     y = y.reshape(-1, 1)

#     sgdreg = SGDRegressor(max_iter=200, penalty=None, eta0=0.1, alpha=0, tol=10e-10, shuffle=False, learning_rate="constant")
#     sgdreg.fit(x,y.ravel(), coef_init=GD.init_beta[1], intercept_init=GD.init_beta[0])
#     beta_sklearn = np.concatenate((sgdreg.intercept_, sgdreg.coef_))

#     msg = f"sklearn: {beta_sklearn}, our: {beta}"
#     assert np.allclose(beta_sklearn, beta), msg

