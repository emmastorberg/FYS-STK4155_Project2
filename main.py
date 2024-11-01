import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

from GradientDescent import Plain, Stochastic
from neural_network import NeuralNetwork
import utils
from utils import sigmoid, sigmoid_der, mse, mse_der, softmax, softmax_der, ReLU, ReLU_der


def main():
    # gd interface

    # cost = mse # callable C(predict, target)
    # cost_der = utils.analytic_grad_OLS # callable, for linear regression: cost_grad(X, y, beta): return lambda beta: {some expression here}
    # optimizer = Stochastic(lr = 0.0001, n_epochs=10000, M=2, momentum=0.3) # should work with Stochastic as well
    # x = np.random.randn(10)
    # X = np.zeros((len(x), 3))
    # X[:,0] = 1
    # X[:,1] = x
    # X[:,2] = x**2

    # y = 3*x**2 + 2*x + 4


    # optimizer.set_gradient(cost_der)
    # beta = [np.random.randn(3)]
    # beta = optimizer.gradient_descent(X, beta, y)
    # print(beta)

    # nn interface

    network_input_size = 4    # int
    layer_output_sizes = [4, 8, 10, 3]  # ints of number of neurons per layer
    activation_funcs = [sigmoid, sigmoid, sigmoid, softmax]    # callable per layer
    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der, softmax_der]
    cost_func = utils.cross_entropy
    cost_der = utils.cross_entropy_der
    optimizer = Stochastic(lr=0.001, M=150, t0=0.1, t1=1, n_epochs=10000)
    # optimizer = Plain(lr = 0.001, max_iter=10000)
    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_func,
        cost_der,
        optimizer,
    )

    inputs, targets = utils.get_iris_data()
    # inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # targets = np.array([[0], [1], [1], [0]])

    nn.train(inputs, targets)
    prediction = nn.predict(inputs)
    print(prediction)
    print(f"accuracy: {utils.accuracy(prediction, targets)}")
    print(prediction)
    print(targets)
    print(f"accuracy: {utils.accuracy(prediction, targets)}")


if __name__ == "__main__":
    main()