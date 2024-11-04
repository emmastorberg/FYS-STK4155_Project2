import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim

from GradientDescent import Plain, Stochastic
from neural_network import NeuralNetwork
from logistic_regression import LogisticRegression
import utils
from utils import sigmoid, sigmoid_der, mse, mse_der, softmax, softmax_der, ReLU, ReLU_der


def pytorch():
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # reshape for a single output
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Define the neural network model (no hidden layers)
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.linear = nn.Linear(input_size, 1)  # Linear layer

        def forward(self, x):
            return torch.sigmoid(self.linear(x))  # Sigmoid activation

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = SimpleNN(input_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Access initial weights and biases
    initial_weights = model.linear.weight.data.clone()
    initial_bias = model.linear.bias.data.clone()

    print("Initial Weights:", initial_weights)
    print("Initial Bias:", initial_bias)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Manual gradient computation (for debugging)
        manual_grad_weights = model.linear.weight.grad.data.clone()
        manual_grad_bias = model.linear.bias.grad.data.clone()

        # Update weights
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Access final weights and biases
    final_weights = model.linear.weight.data.clone()
    final_bias = model.linear.bias.data.clone()

    print("Final Weights:", final_weights)
    print("Final Bias:", final_bias)

    # Access final gradients
    final_grad_weights = model.linear.weight.grad.data.clone()
    final_grad_bias = model.linear.bias.grad.data.clone()

    print("Final Gradients Weights:", final_grad_weights)
    print("Final Gradients Bias:", final_grad_bias)

    # Evaluation on test set
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100
        print(f'Accuracy on test set: {accuracy:.2f}%')


def main():
    data_set = "iris" # "cancer", "iris" or "heart"
    if data_set == "cancer":
        network_input_size = 30
        # layer_output_sizes = [8, 50, 50, 1] # [8, 10, 6, 1]
        n_layers = 3
        layer_output_sizes = [15] * (n_layers-1) + [1]
        # activation_funcs = [sigmoid, sigmoid, sigmoid, sigmoid]
        activation_funcs = [sigmoid] * n_layers
        activation_ders = [sigmoid_der] * n_layers
        # activation_ders = [grad(act) for act in activation_funcs]
        # activation_ders = [sigmoid_der] * n_layers
        cost_func = utils.cross_entropy
        cost_der = utils.cross_entropy_der
        #cost_der = grad(cost_func, 0)

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
    inputs, targets = utils.get_iris_data()

    # new_targets = np.empty((len(targets), 2))
    # for i, target in enumerate(targets):
    #     if target == 1:
    #         new_targets[i, :] = [1, 0]
    #     else:
    #         new_targets[i, :] = [0, 1]
    
    # targets = new_targets
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    nn.train(x_train, y_train)
    prediction = nn.predict(x_train)
    for pred, true in zip(prediction, y_train):
        print(pred, true)
    # print(prediction)
    print(f"accuracy: {utils.accuracy(prediction, y_train)}")


if __name__ == "__main__":
    main()
    # pytorch()