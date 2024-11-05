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


class NN(nn.Module):
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
    cost_func = nn.BCELoss()
    lr = 0.1
    accuracy_dict = {
        'n_hidden_layers': [],
        'n_nodes_pr_layer': [],
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

                accuracy_dict['num_hidden_layers'].append(num_hidden_layers)
                accuracy_dict['node_size'].append(node_size)
                accuracy_dict['accuracy'].append(accuracy)

    df_accuracy = pd.DataFrame(accuracy_dict)

    heatmap_data = df_accuracy.pivot(index='num_hidden_layers', columns='node_size', values='accuracy')
    return heatmap_data


def test_activation_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size, node_size = 40, num_hidden_layers = 1):
    criterion = nn.BCELoss()
    learning_rate = 0.1
    accuracy_dict = {
        'activation_func': [],
        'epoch': [],
        'accuracy': []
    }

    for activation_func in ["sigmoid", "leaky_relu", "relu"]: 
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
        
            accuracy_dict['epoch'].append(epoch)
            accuracy_dict['activation_func'].append(activation_func)
            accuracy_dict['accuracy'].append(accuracy)

    df_accuracy = pd.DataFrame(accuracy_dict)
    heatmap_data = df_accuracy.pivot(index='epoch', columns='activation_func', values='accuracy')
    return heatmap_data

def plot_grid_heatmap(df, save=False, annot = True):
    sns.heatmap(df, annot=annot, cmap='coolwarm')
    plt.title(r"Accuracy for Number of Hidden Layers and Nodes, with $\eta = 0.1$")
    plt.tight_layout()
    if save:
        plt.savefig("figures/all_plots/grid_search_layer_nodes.png")
    else:
        plt.show()

        
def pytorch():
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

    node_size = 5
    num_hidden_layers = 5
    activation_func = torch.sigmoid
    model = LogReg(input_size)
    #model = NN(input_size, node_size, num_hidden_layers, activation_func)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
    plt.title("Confusion Matrix for Breast Cancer Predictions Using PyTorch")
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
        plt.plot(activation_df.index, activation_df[activation], marker='o', label=activation)
    plt.legend()

    if save:
        plt.savefig("figures/all_plots/activationfunctions_cost.png")
    else:
        plt.show()


if __name__ == "__main__":
    y_test, predicted, accuracy_df, activation_df = pytorch()
    #plot_cm(y_test, predicted, save=False)
    plot_grid_heatmap(accuracy_df, save=True)
    plot_activation(activation_df, save=True)
    

