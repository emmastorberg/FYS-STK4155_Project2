import numpy as np
import matplotlib.pyplot as plt

from GradientDescent import PlainFixed, StochasticFixed


def main():
    n = 20
    rng = np.random.default_rng(10)
    x = np.linspace(0, 1, n)
    y = 2*x
    X = np.c_[np.ones(n), x]

    GD = PlainFixed(eta=0.2, rng=rng)
    GD.set_gradient(X, y)
    beta = GD.perform()

    GDA = PlainFixed(eta=0.2, eta_tuner="adagrad", rng=rng)
    GDA.set_gradient(X, y)
    betagda = GD.perform()

    GDM = PlainFixed(eta=0.2,delta_momentum=0.3, rng=rng)
    GDM.set_gradient(X, y)
    betam = GDM.perform()

    GDMA = PlainFixed(eta=0.2, eta_tuner="adagrad",delta_momentum=0.3, rng=rng)
    GDMA.set_gradient(X, y)
    betagdma = GDMA.perform()

    SGD = StochasticFixed(eta=0.2, t0=1, t1=10, rng=rng)
    SGD.set_gradient(X, y)
    betasgd = SGD.perform()

    SGDM = StochasticFixed(eta=0.02, delta_momentum=0.3, t0=0.1, t1=1, rng=rng)
    SGDM.set_gradient(X, y)
    betasgdm = SGDM.perform()

    beta_linreg = np.linalg.pinv(X.T @ X) @ X.T @ y

    xnew = np.array([[0],[2]])
    xbnew = np.c_[np.ones((2,1)), xnew]
    ypredictsgd = xbnew.dot(betasgd)
    ypredictm = xbnew.dot(betam)
    ypredict = xbnew.dot(beta)
    ypredictgda = xbnew.dot(betagda)
    ypredictgdma = xbnew.dot(betagdma)
    ypredictsgdm = xbnew.dot(betasgdm)
    ypredict2 = xbnew.dot(beta_linreg)
    #plt.plot(xnew, ypredictsgdm, label="sgdm")
    #plt.plot(xnew, ypredictsgd, label = "sgd")
    plt.plot(xnew, ypredictm, label="gdm")
    plt.plot(xnew, ypredict, "r-", label="GD")
    plt.plot(xnew, ypredictgda, "c-", label="GD with adagrad")
    plt.plot(xnew, ypredictgdma, "d-", label="GD with adagrad and moment")
    #plt.plot(xnew, ypredict2, "b-", label="analytical")
    plt.plot(x, y ,'ro')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Gradient descent example')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()