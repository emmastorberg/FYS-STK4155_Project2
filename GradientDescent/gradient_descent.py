
import numpy as np


class GradientDescent:
    def __init__(self):
        self.gradient = None
        
    def set_gradient(self, gradient: np.ndarray) -> None:
        """
        Setter method??
        """
        self.gradient = gradient

    def calculate_gradient(self):
        ...
