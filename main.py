import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss

from GradientDescent import Plain, Stochastic
from neural_network import NeuralNetwork
from logistic_regression import LogisticRegression
import utils
from utils import sigmoid, sigmoid_der, mse, mse_der, softmax, softmax_der, ReLU, ReLU_der


def main():
    data_set = "cancer" # "cancer", "iris" or "heart"
    if data_set == "cancer":
        network_input_size = 30
        n_layers = 3
        layer_output_sizes = [15] * (n_layers-1) + [1]
        activation_funcs = [ReLU] * (n_layers - 1) + [sigmoid]
        activation_ders = [ReLU_der] * (n_layers - 1) + [sigmoid_der]
        cost_func = utils.cross_entropy
        cost_der = utils.cross_entropy_der

    elif data_set == "iris":
        network_input_size = 4
        layer_output_sizes = [8, 10, 6, 3]
        activation_funcs = [sigmoid, sigmoid, sigmoid, softmax]
        activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der, softmax_der]
        cost_func = utils.cross_entropy
        cost_der = grad(utils.cross_entropy, 0)

    elif data_set == "heart":
        network_input_size = 8
        n_layers = 3
        layer_output_sizes = [25] * (n_layers-1) + [2]
        activation_funcs = [sigmoid] * n_layers
        activation_ders = [sigmoid_der] * n_layers
        cost_func = utils.binary_cross_entropy
        cost_der = grad(cost_func, 0)

    # optimizer = Stochastic(lr=0.001, M=10, n_epochs=1000, lr_schedule="linear", tuner="adam")
    optimizer = Plain(lr=0.001, max_iter=10000)
    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_func,
        cost_der,
        optimizer,
        seed=18,
    )
    inputs, targets = utils.get_cancer_data()
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    nn.train(x_train, y_train)
    prediction = nn.predict(x_train)
    print(f"accuracy: {utils.accuracy(prediction, y_train)}")


if __name__ == "__main__":
    main()