from neural_network import NeuralNetwork
from utils import sigmoid, sigmoid_der, cross_entropy, cross_entropy_der

class LogisticRegression(NeuralNetwork):
    def __init__(
            self,
            input_size, 
            output_size,
            optimizer,
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
            optimizer
            )

    def train():
        ...

    def predict():
        ...
