# import pytest # type: ignore

import numpy as np
from sklearn.linear_model import SGDRegressor # type: ignore
from autograd import grad

from GradientDescent import Plain, Stochastic
from utils import sigmoid, ReLU, sigmoid_der, ReLU_der, mse, mse_der, cross_entropy
from neural_network import NeuralNetwork

def feed_forward(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def cost(layers, inputs, activation_funcs, target):
    predict = feed_forward(inputs, layers, activation_funcs)
    return mse(predict, target)

def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size).T
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def test_simple_test():
    assert 1 + 1 == 2

def test_backpropagation():
    number_of_datapoints = np.random.randint(2, 20)
    network_input_size = np.random.randint(2, 20)
    final_output_size = np.random.randint(2, 20)

    inputs = np.random.rand(number_of_datapoints, network_input_size)
    layer_output_sizes = [5, 2, final_output_size]
    activation_funcs = [sigmoid, ReLU, sigmoid]
    activation_ders = [sigmoid_der, ReLU_der, sigmoid_der]

    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_func = mse,
        cost_grad = mse_der,
        optimizer=Plain(),
    )

    target = np.random.rand(number_of_datapoints, final_output_size)
    layers = create_layers(network_input_size, layer_output_sizes)
    layer_grads = nn._backpropagation(inputs, target)

    print("Number of datapoints:", number_of_datapoints)
    print("Network input size:", network_input_size)
    print("Final output size:", final_output_size)

    print("Our gradients:")
    for i in range(len(layer_grads)):
        print(i, layer_grads[i][1])

    print("Autograd:")
    cost_grad = grad(cost, 0)
    w_autograd = cost_grad(layers, inputs, activation_funcs, target)
    for i in range(len(w_autograd)):
        print(i, w_autograd[i][1])


if __name__ == "__main__":
    test_backpropagation()


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

