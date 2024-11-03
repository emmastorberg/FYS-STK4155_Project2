from typing import Optional

import autograd.numpy as np

from neural_network import NeuralNetwork
from utils import sigmoid, sigmoid_der, cross_entropy, cross_entropy_der

class LogisticRegression(NeuralNetwork):
    def __init__(
            self,
            input_size, 
            output_size,
            optimizer,
            seed: Optional[int] = None,
            ):
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
        
    def train(self, input, target):
        return super().train(input, target)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return super().predict(inputs)

