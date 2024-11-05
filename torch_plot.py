import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

import utils
torch.manual_seed(2002) 
utils.aesthetic_2D()

class NN(nn.Module):
    """
    A feedforward neural network model for binary classification.

    Attributes:
        output (nn.Linear): Output layer, producing a single value (binary classification).
        layers (nn.ModuleList): List of hidden layers, each a fully connected layer.
        activation_func (nn.Module): Activation function applied to the hidden layers.
    
    Args:
        input_size (int): The number of input features.
        node_size (int): The number of neurons in each hidden layer.
        num_hidden_layers (int): The number of hidden layers in the network.
        activation (str): The activation function to use in hidden layers. 
                          Options are 'relu', 'leaky_relu', 'sigmoid'. 
    
    Raises:
        ValueError: If the provided activation function is not one of the recognized types ('relu', 'leaky_relu', 'sigmoid').

    Methods:
        forward(x):
            Passes input through the network, returning the predicted output.
    """
    def __init__(self, input_size, node_size, num_hidden_layers, activation):
        super(NN, self).__init__()

        self.output = nn.Linear(node_size, 1) #output
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, node_size))  #Input
        self.activation_func = None 

        if activation == 'relu':
            self.activation_func = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_func = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        else:
            raise ValueError

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(node_size, node_size))

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_func(layer(x))

        return torch.sigmoid(self.output(x))
        

class LogReg(nn.Module):
    def __init__(self, input_size):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)) 
    

def gridsearch_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size):
    """
    Performs a grid search to tune the hyperparameters of a neural network

    Args:
        X_train_tensor (Tensor): The training feature data (input) in tensor format.
        X_test_tensor (Tensor): The test feature data (input) in tensor format.
        y_train_tensor (Tensor): The training labels (output) in tensor format.
        y_test_tensor (Tensor): The test labels (output) in tensor format.
        input_size (int): The number of features in the input data.

    Returns:
        pd.DataFrame: A DataFrame containing the accuracy results for each combination of the number of
                      hidden layers and the number of nodes per hidden layer, pivoted for heatmap plotting.
    
    Note:
        - The grid search only considers 1 activation function, 'leaky_relu', because it has previously performed best
        - The model is trained for 50 epochs with stochastic gradient descent (SGD) and a learning rate of 0.1.
    """
    cost_func = nn.BCELoss()
    lr = 0.1
    accuracy_dict = {
        'Number of Hidden Layers': [],
        'Number of Nodes per Hidden Layer': [],
        'accuracy': []
    }

    for n_nodes in range(1, 101, 10):
        for n_hidden_layers in [1, 2, 3, 4, 5]:
            for activation_func in ["leaky_relu"]:
                model = NN(input_size, n_nodes, n_hidden_layers, activation_func)

                n_epochs = 50
                for _ in range(n_epochs):
                    outputs = model(X_train_tensor)
                    cost = cost_func(outputs, y_train_tensor)
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                with torch.no_grad():
                      test_outputs = model(X_test_tensor)
                      predicted = (test_outputs > 0.5).float()
                      accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100

                # Append results to the dictionary
                accuracy_dict['Number of Hidden Layers'].append(n_hidden_layers)
                accuracy_dict['Number of Nodes per Hidden Layer'].append(n_nodes)
                accuracy_dict['accuracy'].append(accuracy)

    df_accuracy = pd.DataFrame(accuracy_dict)

    # Pivot the DataFrame for heatmap plotting
    heatmap_data = df_accuracy.pivot(index='Number of Hidden Layers', columns='Number of Nodes per Hidden Layer', values='accuracy')

    return heatmap_data


def test_activation_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size, node_size = 41, num_hidden_layers = 1):
    """
    Tests different activation functions 'sigmoid', 'relu', and 'leaky_relu'

    The model is trained using stochastic gradient descent (SGD) with a fixed learning rate of 0.1 for 50 epochs.

    Args:
        X_train_tensor (Tensor): The input features for the training set.
        X_test_tensor (Tensor): The input features for the test set.
        y_train_tensor (Tensor): The labels for the training set.
        y_test_tensor (Tensor): The labels for the test set.
        input_size (int): The number of features in the input data.
        node_size (int, optional): The number of nodes in each hidden layer. Defaults to 41.
        num_hidden_layers (int, optional): The number of hidden layers in the network. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame containing accuracy results for each activation function across epochs (for plotting)
    """
    criterion = nn.BCELoss()
    learning_rate = 0.1
    accuracy_dict = {
        'Activation Function': [],
        'Epochs': [],
        'accuracy': []
    }

    for activation_func in ["sigmoid", "relu", "leaky_relu"]: 
        model = NN(input_size, node_size, num_hidden_layers, activation_func)

        num_epochs = 50
        for epoch in range(num_epochs):
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    predicted = (test_outputs > 0.5).float()
                    accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100
        
            # Append results to the dictionary
            accuracy_dict['Epochs'].append(epoch)
            accuracy_dict['Activation Function'].append(activation_func)

            accuracy_dict['accuracy'].append(accuracy)

    df_accuracy = pd.DataFrame(accuracy_dict)

    # Pivot the DataFrame for heatmap plotting
    heatmap_data = df_accuracy.pivot(index='Epochs', columns='Activation Function', values='accuracy')
    return heatmap_data

def plot_grid_heatmap(df, save=False, annot = True, title=r"Accuracy for Number of Hidden Layers and Nodes, with $\eta = 0.1$"):
    sns.heatmap(df, annot=annot)
    plt.title(title)

    plt.tight_layout()
    if save:
        plt.savefig("figures/all_plots/grid_search_layer_nodes.png")
    else:
        plt.show()

        
def pytorch():
    """
    Runs a pipeline for training a neural network on the breast cancer dataset.

    The process includes:
    - Loading and splitting the data.
    - Scaling the features using standardization.
    - Running grid search for hyperparameter optimization.
    - Running tests for different activation functions.
    - Training a final neural network model based on selected hyperparameters.
    - Evaluating the model and returning the predictions and accuracy.

    Args:
        None: All necessary data is loaded and processed within the function.

    Returns:
        tuple: A tuple containing:
            - y_test (ndarray): The true labels for the test set.
            - predicted (ndarray): The predicted labels for the test set from the final trained model.
            - grid_df (pd.DataFrame): The results from the grid search, including accuracy for different hyperparameter configurations.
            - activation_df (pd.DataFrame): The results from testing different activation functions, including accuracy for each function across epochs.
    
    Note:
        - The model is trained using a fixed set of hyperparameters based on the grid search results, specifically:
          `node_size=41`, `num_hidden_layers=1`, and `activation_func="leaky_relu"`.
        - The final model uses Binary Cross-Entropy (BCELoss) and Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
        - The grid search and activation function testing are run for 50 epochs for computational efficiency, while the final model is trained for 100 epochs.
    """
    X, y = utils.get_cancer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    input_size = X_train.shape[1]
    grid_df = gridsearch_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size)
    activation_df = test_activation_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size)

    # Final model based on grid search
    node_size = 41
    num_hidden_layers = 1
    activation_func = "leaky_relu"
    model = NN(input_size, node_size, num_hidden_layers, activation_func)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    learning_rate = 0.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # For comparison with logreg
    #model= LogReg(input_size)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100

    return y_test, predicted, grid_df, activation_df


def plot_cm(target: np.array, predicted: np.array, save = False):
    cf = confusion_matrix(target, predicted)
    group_labels = ['True Neg', 'False Pos','False Neg','True Pos']

    group_counts = ['{0:0.0f}'.format(value) for value in cf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_labels,group_percentages)]
    labels = np.array(labels).reshape(2,2)
    sns.heatmap(cf, annot=labels, fmt="")
    plt.title("Confusion Matrix for Breast Cancer Predictions")
    if save:
        plt.savefig("figures/confusionmatrix.png")
    else:
        plt.show()

def plot_activation(activation_df: pd.DataFrame, save=False):
    plt.figure(figsize=(10, 6))
    plt.title(r"Accuracy for Different Activation Functions in the Hidden Layers, with $\eta = 0.1$")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    for activation in activation_df.columns:
        if activation == "relu":
            label="ReLU"
        elif activation == "sigmoid":
            label="Sigmoid"
        elif activation == "leaky_relu":
            label="Leaky ReLU"

        plt.plot(activation_df.index, activation_df[activation], label=label)
    plt.legend()

    if save:
        plt.savefig("figures/all_plots/activationfunctions_cost.png")
    else:
        plt.show()


if __name__ == "__main__":
    y_test, predicted, accuracy_df, activation_df = pytorch()
    plot_cm(y_test, predicted, save=False) 
    plot_grid_heatmap(accuracy_df, save=False)
    plot_activation(activation_df, save=False)
    

