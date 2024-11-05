import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from plotroutines import aesthetic_2D
import pandas as pd

torch.manual_seed(2002) 
aesthetic_2D()

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

            return torch.sigmoid(self.output(x))  # Sigmoid activation for the output layer
        


# Define the neural network model (no hidden layers)
class LogReg(nn.Module):
    def __init__(self, input_size):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Linear layer

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation
    
def gridsearch_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size):
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    learning_rate = 0.1
    accuracy_dict = {
        'Number of Hidden Layers': [],
        'Number of Nodes per Hidden Layer': [],
        'accuracy': []
    }

    for node_size in range(1, 100, 10):
        for num_hidden_layers in [1, 2, 3, 4, 5]:#, 6, 7, 8, 9, 10]:
            for activation_func in ["leaky_relu"]: # nn.LeakyReLU
                model = NN(input_size, node_size, num_hidden_layers, activation_func)

                # Training loop
                num_epochs = 50
                for epoch in range(num_epochs):

                    # Forward pass
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)

                    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    # Print loss every 10 epochs
                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

                    
                with torch.no_grad():
                      test_outputs = model(X_test_tensor)
                      predicted = (test_outputs > 0.5).float()
                      accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100
                
                # Append results to the dictionary
                accuracy_dict['Number of Hidden Layers'].append(num_hidden_layers)
                accuracy_dict['Number of Nodes per Hidden Layer'].append(node_size)
                accuracy_dict['accuracy'].append(accuracy)

    # Create DataFrame from the dictionary
    df_accuracy = pd.DataFrame(accuracy_dict)

    # Pivot the DataFrame for heatmap plotting
    heatmap_data = df_accuracy.pivot(index='Number of Hidden Layers', columns='Number of Nodes per Hidden Layer', values='accuracy')
    return heatmap_data


def test_activation_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size, node_size = 40, num_hidden_layers = 1):
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
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

            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    predicted = (test_outputs > 0.5).float()
                    accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100
        
            # Append results to the dictionary
            accuracy_dict['Epochs'].append(epoch)
            accuracy_dict['Activation Function'].append(activation_func)
            accuracy_dict['accuracy'].append(accuracy)

    # Create DataFrame from the dictionary
    df_accuracy = pd.DataFrame(accuracy_dict)

    # Pivot the DataFrame for heatmap plotting
    heatmap_data = df_accuracy.pivot(index='Epochs', columns='Activation Function', values='accuracy')
    return heatmap_data

def plot_grid_heatmap(df, save=False, annot = True, title=r"Accuracy for Number of Hidden Layers and Nodes, with $\eta = 0.1$"):
    sns.heatmap(df, annot=annot)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig("figures/grid_search_layer_nodes.png")
    else:
        plt.show()

        
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

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]

    grid_df = gridsearch_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size)

    activation_df = test_activation_pytorch(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size)

    node_size = 5
    num_hidden_layers = 5
    activation_func = torch.sigmoid
    model = LogReg(input_size)
    model = NN(input_size, node_size, num_hidden_layers, activation_func)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
        # manual_grad_weights = model.linear.weight.grad.data.clone()
        # manual_grad_bias = model.linear.bias.grad.data.clone()

        # Update weights
        optimizer.step()

        # Print loss every 10 epochs
        #if (epoch + 1) % 10 == 0:
            #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


        # Evaluation on test set
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100
            #print(f'Accuracy on test set: {accuracy:.2f}%')

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
        plt.savefig("figures/activationfunctions_cost.png")
    else:
        plt.show()


if __name__ == "__main__":
    y_test, predicted, accuracy_df, activation_df = pytorch()
    #plot_cm(y_test, predicted, save=False)
    plot_grid_heatmap(accuracy_df, save=True)
    plot_activation(activation_df, save=True)
    

