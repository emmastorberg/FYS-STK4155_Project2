import numpy as np
import matplotlib.pyplot as plt

from GradientDescent import PlainFixed, PlainMomentum


def main():
    n = 20
    x = np.linspace(0, 1, n)
    y = 2*x
    X = np.c_[np.ones(n), x]

    GD = PlainFixed(0.1, beta_len=2, max_iter=200, seed=8)
    gradient = lambda beta: (2.0/n)*X.T @ (X @ beta-y)
    GD.set_gradient(gradient)
    beta = GD.perform()
    print(beta)

    # GDM = PlainMomentum(0.1, beta_len=2, max_iter=200)
    # GDM.set_gradient(gradient)
    # betam = GDM.perform()

    # beta_linreg = np.linalg.pinv(X.T @ X) @ X.T @ y

    # xnew = np.array([[0],[2]])
    # xbnew = np.c_[np.ones((2,1)), xnew]
    # ypredictm = xbnew.dot(betam)
    # ypredict = xbnew.dot(beta)
    # ypredict2 = xbnew.dot(beta_linreg)
    # plt.plot(xnew, ypredictm, label="gdm")
    # plt.plot(xnew, ypredict, "r-", label="GD")
    # plt.plot(xnew, ypredict2, "b-", label="analytical")
    # plt.plot(x, y ,'ro')
    # plt.axis([0,2.0,0, 15.0])
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$y$')
    # plt.title(r'Gradient descent example')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()