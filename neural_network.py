import itertools
from typing import Optional, Callable

import autograd.numpy as np # type: ignore

from GradientDescent import GD


class NeuralNetwork:
    """A class for implementing neural networks."""
    def __init__(
            self,
            network_input_size: int,
            layer_output_sizes: list[int],
            activation_funcs: list[str],
            activation_ders: list[str],
            cost_func: Callable,
            cost_der: Callable,
            optimizer: GD,
            seed: Optional[int] = None
        ) -> None:
        """Initialize the nerual network.

        Args:
            network_input_size (int): Number of features.
            layer_output_sizes (list[int]): 
                A list containing number of nodes for each layer. 
                One element in the list corresponds to a single layer.
            activation_funcs (list[str]): List of callable activation functions for each layer.
            activation_ders (list[str]): List of callable derivatives of activation functions for each layer.
            cost_func (Callable): Callable cost function.
            cost_der (Callable): Callable derivative of cost function.
            optimizer (GD): Instance of a gradient descent method.
            seed (Optional[int], optional): Random seed. Defaults to None.
        """
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_func = cost_func
        self.cost_der = cost_der
        self.optimizer = optimizer
        self.seed = seed

        self.layers = self.create_layers(network_input_size, layer_output_sizes)

    def create_layers(self, network_input_size: int, layer_output_sizes: list[int]) -> list:
        """Create layers of weights and biases.

        Args:
            network_input_size (int): Number of features.
            layer_output_sizes (list[int]): 
                A list containing number of nodes for each layer. 
                One element in the list corresponds to a single layer.

        Returns:
            list: List of tuples with weights and biases for each layer. 
        """
        np.random.seed(self.seed)
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            std = np.sqrt(2 / (layer_output_size + i_size))
            W = np.random.normal(scale=std, size=(i_size, layer_output_size))
            b = np.zeros(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size

        return layers

    @staticmethod
    def feed_forward(inputs: np.ndarray, layers: list, activation_funcs: list) -> np.ndarray:
        """Signle feed forward pass.

        Args:
            inputs (np.ndarray): Input values.
            layers (list): List of tuples with weights and biases for each layer. 
            activation_funcs (list): List of callable activation functions for each layer.

        Returns:
            np.ndarray: Predicted values.
        """
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def feed_forward_saver(self, input: np.ndarray) -> tuple:
        """Single feed forward pass, and save layer inputs, and return prediction.

        Args:
            input (np.ndarray): Input values.

        Returns:
            tuple:
                layers inputs, layer outputs without activation, and prediction.
        """
        layer_inputs = []
        zs = []
        a = input
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    def backpropagation(self, input: np.ndarray, target: np.ndarray) -> list:
        """Perform backpropagation.

        Args:
            input (np.ndarray): Input values.
            target (np.ndarray): Target values.

        Returns:
            list: List of tuples containing the derivative of the cost with respect to the corresponing weights and biases.
        """
        layer_inputs, zs, predict = self.feed_forward_saver(input)
        num_layers = len(self.layers)
        layer_grads = [None] * num_layers

        for i in reversed(range(num_layers)):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == num_layers - 1:
                dC_da = self.cost_der(predict, target)
            else:
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T
            dC_dz = activation_der(z, dC_da)
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0) * len(layer_input)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    def flatten_layers(self, layers: list) -> list:
        """Flatten layers for list of tuples to a flat list.

        Args:
            layers (list): List of typles of weights and biases.

        Returns:
            list: Flat list of weights and biases.
        """
        return list(itertools.chain(*layers))
    
    def reconstruct_layers(self, layers: list) -> list:
        """Reconstruct a flat list of weights and biases back to a list of tuples.

        Args:
            layers (list): Flat list of weights and biases.

        Returns:
            lsit: List of tuples.
        """
        flat_to_tuple = []
        for i in range(len(layers) // 2):
            flat_to_tuple.append((layers[2*i], layers[2*i+1])) 
        return flat_to_tuple
    
    def gradient(self, input: np.ndarray, layers: list, target: np.ndarray) -> list:
        """Compute the gradient of the cost for specified inputs, layers and target.

        Args:
            input (np.ndarray): Input values.
            layers (lits): Flat list of weights and biases.
            target (np.ndarray): Target values.

        Returns:
            list: Flattened list containing the derivative of the cost with respect to the corresponing weights and biases.
        """
        layers = self.reconstruct_layers(layers)
        self.layers = layers
        layer_grads = self.backpropagation(input, target)
        layer_grads = self.flatten_layers(layer_grads)
        
        return layer_grads

    def train(self, input: np.ndarray, target: np.ndarray) -> None:
        """Train the neural network.

        Args:
            input (np.ndarray): Input values.
            target (np.ndarray): Target values.
        """
        self.optimizer.set_gradient(self.gradient)
        layers = self.flatten_layers(self.layers)
        layers = self.optimizer.gradient_descent(input, layers, target)
        self.layers = self.reconstruct_layers(layers)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict for given input values.

        Args:
            inputs (np.ndarray): Input values.

        Returns:
            np.ndarray: Prediction.
        """
        return self.feed_forward(inputs, self.layers, self.activation_funcs)
    
    def cost(self, inputs, target):
        prediction = self.predict(inputs)
        return self.cost_func(prediction, target)
