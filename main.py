import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from GradientDescent import Plain, Stochastic
from neural_network import NeuralNetwork
from logistic_regression import LogisticRegression
import utils
from utils import sigmoid, sigmoid_der, mse, mse_der, softmax, softmax_der, ReLU, ReLU_der

import torch
import torch.nn as nn
import torch.optim as optim

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

    network_input_size = 30 
    layer_output_sizes = [20, 1]
    activation_funcs = [ReLU, sigmoid, sigmoid, sigmoid]
    activation_ders = [ReLU_der, sigmoid_der, sigmoid_der, sigmoid_der]
    cost_func = utils.binary_cross_entropy
    cost_der = grad(utils.binary_cross_entropy, 0)
    # optimizer = Stochastic(lr=0.1, M=5, n_epochs=1000, lr_schedule="linear", tuner="adam")
    optimizer = Plain(lr=0.1, max_iter=10000)
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
    print(prediction)
    print(f"accuracy: {utils.accuracy(prediction, y_train)}")


if __name__ == "__main__":
    main()
    # pytorch()