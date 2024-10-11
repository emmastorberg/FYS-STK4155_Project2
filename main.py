import numpy as np
import matplotlib.pyplot as plt

from GradientDescent import PlainFixed, PlainMomentum, StochasticFixed


def main():
    n = 20
    rng = np.random.default_rng(10)
    x = np.linspace(0, 1, n)
    y = 2*x
    X = np.c_[np.ones(n), x]

    GD = PlainFixed(0.1, beta_len=2, max_iter=200, rng=rng)
    gradient = lambda beta: (2.0/n)*X.T @ (X @ beta-y)
    GD.set_gradient(X, y)
    beta = GD.perform()

    GDM = PlainMomentum(0.1, beta_len=2, max_iter=200,rng=rng)
    GDM.set_gradient(X, y)
    betam = GDM.perform()

    SGD = StochasticFixed(0.1, beta_len=2, max_iter=200, rng=rng)
    SGD.set_gradient(X, y)
    betasgd = SGD.perform()

    SGDM = StochasticFixed(0.1, beta_len=2, max_iter=200, rng=rng)
    SGDM.set_gradient(X, y)
    betasgdm = SGDM.perform()

    beta_linreg = np.linalg.pinv(X.T @ X) @ X.T @ y

    xnew = np.array([[0],[2]])
    xbnew = np.c_[np.ones((2,1)), xnew]
    ypredictsgd = xbnew.dot(betasgd)
    ypredictm = xbnew.dot(betam)
    ypredict = xbnew.dot(beta)
    ypredictsgdm = xbnew.dot(betasgdm)
    ypredict2 = xbnew.dot(beta_linreg)
    plt.plot(xnew, ypredictsgdm, label="sgdm")
    plt.plot(xnew, ypredictsgd, label = "sgd")
    plt.plot(xnew, ypredictm, label="gdm")
    plt.plot(xnew, ypredict, "r-", label="GD")
    plt.plot(xnew, ypredict2, "b-", label="analytical")
    plt.plot(x, y ,'ro')
    plt.axis([0,2.0,0, 15.0])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Gradient descent example')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()