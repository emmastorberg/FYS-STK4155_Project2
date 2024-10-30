import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

from GradientDescent import Plain, Stochastic
from neural_network import NeuralNetwork
import utils
from utils import sigmoid, sigmoid_der, mse, mse_der, softmax, softmax_der, ReLU, ReLU_der


def main():
    # gd interface

    cost = mse # callable C(predict, target)
    cost_grad = utils.analytic_grad_OLS # callable, for linear regression: cost_grad(X, y, beta): return lambda beta: {some expression here}
    optimizer = Plain(lr = 0.01, max_iter=10000) # should work with Stochastic as well
    x = np.random.randn(10).reshape(-1, 1)
    # X = np.zeros((len(x), 3))
    # X[:,0] = 1
    # X[:,1] = x
    # X[:,2] = x**2

    y = 3*x**2 + 2*x + 4


    # optimizer.set_gradient(cost_grad)
    # beta = [np.random.randn(3)]
    # beta = optimizer.gradient_descent(X, beta, y)
    # print(beta)

    # nn interface

    network_input_size = 4    # int
    layer_output_sizes = [8, 10, 3]  # ints of number of neurons per layer
    activation_funcs = [sigmoid, sigmoid, sigmoid, utils.no_activation]    # callable per layer
    activation_grads = [sigmoid_der, sigmoid, sigmoid, utils.no_activation_der]
    cost_func = utils.cross_entropy
    cost_grad = grad(utils.cross_entropy, 0)
    optimizer = Plain(max_iter=1000, lr=0.1)
    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_grads,
        cost_func,
        cost_grad,
        optimizer,
    )

    inputs, targets = utils.get_iris_data()

    nn.train(inputs, targets)
    prediction = nn.predict(inputs)
    print(prediction)
    print(targets)
    # print(f"accuracy: {utils.accuracy(prediction, targets)}")


if __name__ == "__main__":
    main()