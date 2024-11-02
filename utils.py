import autograd.numpy as np # type: ignore
from autograd import jacobian
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score


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

def ReLU_der(z):
    der = np.where(z > 0, 1, 0)
    return np.stack([np.diag(row) for row in der])

def leaky_ReLU(z):
    alpha = 0.01
    return np.where(z > 0, z, 0) + np.where(z < 0, alpha * z, 0)

def leaky_ReLU_der(z):
    alpha = 0.01
    der = np.where(z > 0, 1, 0) + np.where(z < 0, alpha, 0)
    return np.stack([np.diag(row) for row in der])

def sigmoid(z):
    # if np.any(np.isinf((1 + np.exp(-z)))):
    #     print("hei")
    neg = np.where(z < 0, np.exp(z) / (1 + np.exp(z)), 0)
    pos = np.where(z >= 0, 1 / (1 + np.exp(-z)), 0)
    return pos + neg

def sigmoid_der(z):
    der = sigmoid(z) * (1 - sigmoid(z))
    return np.stack([np.diag(row) for row in der])

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# def softmax_der(z):
#     gradient = elementwise_grad(softmax, 0)
#     return gradient(z)

def softmax_der(z):
    s = softmax(z)
    return np.stack([np.diag(row) - np.outer(row, row.T) for row in s])

def mse(predict, target):
    n = len(target)
    return (1/n) * np.sum((predict - target)**2)

def mse_der(predict, target):
    n = len(target)
    return (2/n) * (predict - target)

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def binary_cross_entropy(predict, target):
    # Clip predictions to avoid log(0)
    eps = 1e-8
    predict = np.clip(predict, eps, 1 - eps)
    return -np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))

def cross_entropy_der(predict, target):
    # predict = np.clip(predict, 1e-7, 1 - 1e-7)
    return - target / predict

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
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

def analytic_grad_OLS(X, beta, y):
    return [(2.0/len(X)) * X.T @ (X @ beta[0] - y)]

def cost_OLS():
    ...

def cost_ridge():
    ...

class analytic_grad_Ridge:
    def __init__(self, lmbda):
        self.lmbda = lmbda
    
    def __call__(self, X, beta, y):
        return [2.0 * X.T @ (X @ beta[0] - y) + self.lmbda * beta[0]]
    

