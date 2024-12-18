from collections.abc import Iterable


import autograd.numpy as np # type: ignore
from autograd import jacobian, grad
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns


def Franke_function(x: np.ndarray, y: np.ndarray, noise: bool = True) -> np.ndarray:
    """Generates output data from Franke function, with optional noise.

    Args:
        x (np.ndarray): input values
        y (np.ndarray): input values
        noise (bool, optional): Boolean deciding whether or not to make noisy data. Defaults to True.

    Returns:
        np.ndarray: Output after input data is given to Franke function, and noise is possibly applied.
    """
    term1 = 3 / 4 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    term2 = 3 / 4 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    term3 = 1 / 2 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    term4 = -1 / 5 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)

    Franke = term1 + term2 + term3 + term4

    if noise:
        state = np.random.get_state()
        Franke += np.random.normal(0, 0.1, x.shape)
        np.random.set_state(state)

    return Franke

def generate_data_Franke(n: int, seed: int | None = None, noise: bool = True, mesh: bool = False) -> tuple[np.ndarray]:
    """Generates noisy data to be given to our model. Can give multivariate or univariate data.

    Args:
        n (int): number of data points
        seed (int or None): set seed for consistency with random noise
        multidim (bool, optional): Whether or not to make the data multivariate. Defaults to False.

    Returns:
        tuple[np.ndarray]: Input and output data in a tuple. If multivariate, input data is itself 
        a tuple of various inputs X1 and X2.
    """
    np.random.seed(seed)
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    if mesh:
        X1, X2 = np.meshgrid(x1, x2)
        Y = Franke_function(X1, X2, noise)

        x1 = X1.flatten().reshape(-1, 1)
        x2 = X2.flatten().reshape(-1, 1)
        x = (x1, x2)
        y = Y.flatten().reshape(-1, 1)

    else:
        x = np.empty((n, 2))
        y = Franke_function(x1, x2, noise)
        x[:,0] = x1
        x[:,1] = x2

    return x, y

def no_activation(z):
    return z

def no_activation_der(z):
    return np.ones_like(z)

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z, dC_da):
    return dC_da * np.where(z > 0, 1, 0)

def ReLU_jacobi(z):
    der = np.where(z > 0, 1, 0)
    return np.stack([np.diag(row) for row in der])

def leaky_ReLU(z):
    alpha = 0.01
    return np.where(z > 0, z, 0) + np.where(z < 0, alpha * z, 0)

def leaky_ReLU_der(z, dC_da):
    alpha = 0.1
    der = np.where(z > 0, 1, 0) + np.where(z < 0, alpha, 0)
    return dC_da * der

def leaky_ReLU_jacobi(z):
    alpha = 0.01
    der = np.where(z > 0, 1, 0) + np.where(z < 0, alpha, 0)
    return np.stack([np.diag(row) for row in der])

    # if np.any(np.isinf((1 + np.exp(-z)))):
    #     print("hei")
    # eps = 1e-15
    # z = np.clip(z, eps, 1-eps)
    # neg = np.where(z < 0, np.exp(z) / (1 + np.exp(z)), 0)
    # pos = np.where(z >= 0, 1 / (1 + np.exp(-z)), 0)
    # return neg + pos

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    # eps = 1e-15
    # z = np.clip(z, eps, 1-eps)
    # neg = np.where(z < 0, np.exp(z) / (1 + np.exp(z)), 0)
    # pos = np.where(z >= 0, 1 / (1 + np.exp(-z)), 0)
    # return neg + pos

def sigmoid_der(z, dC_da):
    der = sigmoid(z) * (1 - sigmoid(z))
    return dC_da * der

def sigmoid_jacobi(z):
    jacobi = jacobian(sigmoid, 0)
    der = []
    for row in z:
        der.append(jacobi(row))
    der = np.array(der)
    return der

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def softmax_jacobi(z):
    s = softmax(z)
    return np.stack([np.diag(row) - np.outer(row, row.T) for row in s])

def softmax_der(z, dC_da):
    s = softmax(z)
    jacobi = np.stack([np.diag(row) - np.outer(row, row.T) for row in s])
    return np.einsum("ij, ijk -> ik", dC_da, jacobi)
    # return np.stack([np.diag(row) - np.outer(row, row.T) for row in s])

def mse(predict, target):
    n = len(target)
    return (1/n) * np.sum((predict - target)**2)

def mse_der(predict, target):
    n = len(target)
    return (2/n) * (predict - target)

def cross_entropy(predict, target):
    # eps = 1e-8
    # predict = np.clip(predict, eps, 1 - eps)
    return np.sum(-target * np.log(predict))

def binary_cross_entropy(predict, target):
    return -np.mean((target * np.log(predict) + (1 - target) * np.log(1 - predict)))
    # Clip predictions to avoid log(0)
    # eps = 1e-8
    # predict = np.clip(predict, eps, 1 - eps)
    # print(np.max(predict), np.min(predict), np.mean(predict))
    # return -np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))

def binary_cross_entropy_der(predict, target):
    # predict = np.clip(predict, 1e-7, 1 - 1e-7)
    # return - target / predict
    x = -(target * 1 / predict - (1 - target) * 1/(1 - predict)) / predict.size
    
    return np.mean(x, axis=0).reshape(-1, 1)
    return -(target * 1 / predict - (1 - target) * 1/(1 - predict)) / predict.size
    

def get_iris_data():
    iris = load_iris()
    inputs = iris.data
    targets = np.zeros((len(iris.data), 3))
    for i, t in enumerate(iris.target):
        targets[i, t] = 1
    return inputs, targets

def get_cancer_data():
    iris = load_breast_cancer()
    inputs = iris.data
    targets = iris.target
    return inputs, targets

def accuracy(predictions, targets):
    if predictions.shape[1] == 1:
        preds = predictions > 0.5
        return np.mean(preds == targets)
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

def analytic_beta_OLS(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def analytic_grad_OLS(X, beta, y):
    return [(2.0/len(X)) * X.T @ (X @ beta[0] - y)]

def cost_OLS(X, beta, y):
    beta = beta[0]
    return [mse(X @ beta, y)]

class cost_Ridge:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, X, beta, y):
        beta = beta[0]
        return [mse(X @ beta, y) + self.lmbda * (beta.T @ beta)]

class analytic_grad_Ridge:
    def __init__(self, lmbda):
        self.lmbda = lmbda
    
    def __call__(self, X, beta, y):
        beta = beta[0]
        return [2.0 * X.T @ (X @ beta - y) + self.lmbda * beta]

def get_heart_data():
    data = pd.read_csv("data/heart.csv")
    data.Sex.replace({"M": 0, "F": 1}, inplace=True)
    data.ExerciseAngina.replace({"N": 0, "Y": 1}, inplace=True)
    data.drop(["ChestPainType", "RestingECG", "ST_Slope"], axis=1, inplace=True)
    a =data.to_numpy()
    inputs = a[:,:-1]
    targets = a[:,-1]
    return inputs, targets


def aesthetic_2D():
    plt.rcParams.update({
        # Matplotlib style settings similar to seaborn's default style
        "axes.facecolor": "#eaeaf2",
        "axes.edgecolor": "white",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1,
        "axes.axisbelow": True,
        "xtick.color": "gray",
        "ytick.color": "gray",

        # Additional stylistic settings
        "figure.facecolor": "white",
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.fancybox": True,
        "legend.edgecolor": 'lightgray',
    })


if __name__ == "__main__":
    print(get_heart_data())