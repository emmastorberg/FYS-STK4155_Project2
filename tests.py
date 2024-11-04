import pytest
import autograd.numpy as np
from autograd import grad, jacobian
from sklearn.metrics import log_loss

from GradientDescent import Plain
import utils
from neural_network import NeuralNetwork

def feed_forward(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a


def cost_mse(layers, inputs, activation_funcs, target):
    predict = feed_forward(inputs, layers, activation_funcs)
    return utils.mse(predict, target)


def cost_cross_entropy(layers, inputs, activation_funcs, target):
    predict = feed_forward(inputs, layers, activation_funcs)
    return utils.cross_entropy(predict, target)


@pytest.mark.parametrize("func, der", 
                    [
                        (utils.sigmoid, utils.sigmoid_jacobi),
                        (utils.ReLU, utils.ReLU_jacobi),
                        (utils.softmax_vec, utils.softmax_jacobi),
                        (utils.leaky_ReLU, utils.leaky_ReLU_jacobi)
                    ]
                )
def test_autograd_activation(func, der):
    input = np.random.randn(10, 4)
    analytic = der(input)
    gradient = jacobian(func, 0)
    numeric = np.stack([gradient(row) for row in input])
    assert np.allclose(analytic, numeric)


@pytest.mark.parametrize("func, der", 
                    [
                        (utils.mse, utils.mse_der),
                    ]
                )
def test_autgrad_cost(func, der):
    predict = np.random.randn(6, 4)
    target = np.random.randn(4)
    analytic = der(predict, target)
    gradient = grad(func, 0)
    numeric = gradient(predict, target)
    assert np.allclose(analytic, numeric)


def test_leaky_ReLU():
    input = np.array([-2, -1, 0, 1, 2])
    expected = np.array([-0.02, -0.01, 0, 1, 2])
    computed = utils.leaky_ReLU(input)
    assert np.allclose(expected, computed)


def test_backpropagation():
    number_of_datapoints = np.random.randint(2, 20)
    network_input_size = np.random.randint(2, 20)
    final_output_size = np.random.randint(2, 20)

    inputs = np.random.rand(number_of_datapoints, network_input_size)
    layer_output_sizes = [5, 2, final_output_size]
    activation_funcs = [utils.sigmoid, utils.ReLU, utils.sigmoid]
    activation_ders = [utils.sigmoid_der, utils.ReLU_der, utils.sigmoid_der]

    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_func = utils.mse,
        cost_der = utils.mse_der,
        optimizer=Plain(),
    )

    target = np.random.rand(number_of_datapoints, final_output_size)
    layer_grads = nn.backpropagation(inputs, target)

    layers = nn.layers
    cost_grad = grad(cost_mse, 0)
    w_autograd = cost_grad(layers, inputs, activation_funcs, target)
    print(layer_grads[0][1])
    print(w_autograd[0][1])

    for i in range(len(layers)):
        assert np.allclose(layer_grads[i][1], w_autograd[i][1]), f"W, {i}"
        assert np.allclose(layer_grads[i][0], w_autograd[i][0]), f"b, {i}"


def test_iris_data_backprop():
    inputs, target = utils.get_iris_data()
    network_input_size = 4
    layer_output_sizes = [8, 6, 4, 3]
    activation_funcs = [utils.sigmoid, utils.sigmoid, utils.sigmoid, utils.softmax]
    activation_ders = [utils.sigmoid_der, utils.sigmoid_der, utils.sigmoid_der, utils.softmax_der]

    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_func = utils.cross_entropy,
        cost_der = utils.cross_entropy_der,
        optimizer=Plain(),
    )

    layer_grads = nn.backpropagation(inputs, target)

    layers = nn.layers
    cost_der = grad(cost_cross_entropy, 0)
    w_autograd = cost_der(layers, inputs, activation_funcs, target)

    for i in range(len(layers)):
        assert np.allclose(layer_grads[i][1], w_autograd[i][1]), f"{i}, W"
        assert np.allclose(layer_grads[i][0], w_autograd[i][0]), f"{i}, b"


@pytest.mark.parametrize("pred, target", 
                            [
                                (np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1])), 
                                (np.array([0.9, 0.1, 0.8]), np.array([1, 0, 1])),
                                (np.array([0.5, 0.5, 0.5]), np.array([1, 0, 1])),
                            ]
                        )
def test_cross_entropy(pred, target):
    computed = utils.binary_cross_entropy(pred, target)
    expected = log_loss(target, pred)
    msg = f"computed: {computed}, expected: {expected}, pred: {pred}, target: {target}"
    assert np.isclose(computed, expected), msg


def test_gradient_descent():
    layers = []
    for i in range(3):
        W = np.random.randn(5, 8)
        b = np.random.randn(5)
        layers.append(W)
        layers.append(b)

    def gradient(input, layers, target):
        layers_grads = []
        for i, param in enumerate(layers):
            derivative = np.ones_like(param) * 1 / (i+1)
            layers_grads.append(derivative)
        return layers_grads

    inputs = None
    targets = None

    optimizer = Plain(lr=0.01, max_iter=3)
    optimizer.set_gradient(gradient)
    optim_layers = optimizer.gradient_descent(inputs, layers, targets)

    expected = []
    for i in range(3):
        grads = gradient(inputs, layers, targets)
        for param, d_param in zip(layers, grads):
            param -= 0.01 * d_param
            if i == 2:
                expected.append(param)

    for pred, computed in zip(expected, optim_layers):
        assert np.allclose(pred, computed)

def test_flatten_reconstruct():
    tuples = [(1, 2), (3, 4), (5, 6)]
    nn = NeuralNetwork(4, [2], None, None, None, None, None)
    flattened_tuples = nn.flatten_layers(tuples)
    expected = [1, 2, 3, 4, 5, 6]
    assert flattened_tuples == expected

    reconstructed_tuples = nn.reconstruct_layers(flattened_tuples)
    assert reconstructed_tuples == tuples







    

# if __name__ == "__main__":
#     test_gradient_descent()
#     test_backpropagation()
#     test_iris_data_backprop()
#     test_autograd()
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

