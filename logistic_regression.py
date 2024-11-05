from typing import Optional

import autograd.numpy as np

from neural_network import NeuralNetwork
from utils import sigmoid, sigmoid_der, cross_entropy, cross_entropy_der
from GradientDescent import GD

class LogisticRegression(NeuralNetwork):
    """Class for implementing Logistic Regression"""
    def __init__(
            self,
            input_size: int, 
            output_size: int,
            optimizer: GD,
            seed: Optional[int] = None,
            ):
        """Initialize the Logistic Regression Model.

        Args:
            input_size (int): Number of features.
            output_size (list[int]): Number of outputs.
                A list containing number of nodes for each layer. 
                One element in the list corresponds to a single layer.
            optimizer (GD): Instance of a gradient descent method.
            seed (Optional[int], optional): Random seed. Defaults to None.
        """
        output_size = [output_size]
        activation_funcs = [sigmoid]
        activation_ders = [sigmoid_der]
        cost_func = cross_entropy
        cost_der = cross_entropy_der

        super().__init__(
            input_size, 
            output_size, 
            activation_funcs, 
            activation_ders, 
            cost_func, 
            cost_der, 
            optimizer,
            seed,
            )
        
    def train(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Train the neural network.

        Args:
            input (np.ndarray): Input values.
            target (np.ndarray): Target values.
        """
        return super().train(input, target)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict for given input values.

        Args:
            inputs (np.ndarray): Input values.

        Returns:
            np.ndarray: Prediction.
        """
        return super().predict(inputs)

