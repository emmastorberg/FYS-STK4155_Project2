import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

    network_input_size = 30    # int
    layer_output_sizes = [1]  # ints of number of neurons per layer
    activation_funcs = [sigmoid, sigmoid, sigmoid]    # callable per layer
    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der]
    cost_func = utils.binary_cross_entropy
    cost_der = grad(utils.binary_cross_entropy, 0)
    # optimizer = Stochastic(lr=0.001, M=150, t0=0.1, t1=1, n_epochs=10000)
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
    W = np.array([-0.1949, -0.0777, -0.1811, -0.1148, -0.0142, -0.2183, -0.1236, -0.2559,
         -0.0307,  0.0081, -0.0525,  0.0234,  0.0356, -0.0511,  0.1479, -0.0177,
          0.0491, -0.0476, -0.1273, -0.0868, -0.1759, -0.2689, -0.3187, -0.0961,
          0.0590, -0.0105, -0.0658, -0.3271,  0.0525, -0.1866]).reshape(-1, 1)
    
    b = np.array([-0.0488])
    nn.layers = [(W, b)]
    cancer = load_breast_cancer()
    inputs = cancer.data
    targets = cancer.target
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets)
    

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = x_train[:15,:]
    y_train = y_train[:15]
    print(y_train)
    for elem in x_train:
        print(elem)
    # print(nn.backpropagation(x_train, y_train))
    
    # inputs, targets = utils.get_iris_data()

    # # inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # # targets = np.array([[0], [1], [1], [0]])

    # nn.train(x_train, y_train)
    # prediction = nn.predict(x_train)
    # for pred, targ in zip(prediction, y_train):
    #     print(f"prediction: {pred[0]}       target: {targ}")
    # print(utils.cross_entropy(prediction, y_train))
    # print(f"accuracy: {accuracy_score(targets, prediction)}")
    
    # print(f"accuracy: {utils.accuracy(prediction, targets)}")
    # print(prediction)
    # print(targets)
    # print(f"accuracy: {utils.accuracy(prediction, targets)}")
    # cancer = load_breast_cancer()
    # inputs = cancer.data
    # targets = cancer.target

    # logreg = LogisticRegression(30, 1, Stochastic(n_epochs=500, M=3, t0=0.01, t1=10))
    # logreg.train(inputs, targets)
    # layers = logreg.layers
    # prediction = logreg.predict(inputs)
    # print(prediction)
    # print(f"accuracy: {utils.accuracy(prediction, targets)}")


    # new_target = np.empty((len(targets), 2))
    # for i, target in enumerate(targets):
    #     if target == 0:
    #         new_target[i,:] = np.array([1, 0])
    #     else:
    #         new_target[i,:] = np.array([0, 1])

    # targets = new_target



if __name__ == "__main__":
    main()
    # pytorch()