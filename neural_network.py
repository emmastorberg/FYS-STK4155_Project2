import autograd.numpy as np # type: ignore
from sklearn import datasets # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

from activation_funcs import ReLU, signmoid, softmax


class NeuralNetwork:
    def __init__(self, network_input_size: int, layer_output_sizes: list[int]) -> None:
        self.layers = self._create_layers(network_input_size, layer_output_sizes)

    def _create_layers(self, network_input_size: int, layer_output_sizes: list[int]) -> list:
        layers = []

        i_size = network_input_size
        for layer_ouput_size in layer_output_sizes:
            W = np.random.randn(layer_ouput_size, i_size).T
            b = np.random.randn(layer_ouput_size)
            layers.append((W, b))

            i_size = layer_ouput_size
        
        return layers

    def _feed_forward(self, inputs: np.ndarray, layers: list, activation_funcs: list) -> np.ndarray:
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def train(self):
        ...

    def predict(self, inputs: np.ndarray, layers: list, activation_funcs: list) -> np.ndarray:
        return self._feed_forward(inputs, layers, activation_funcs)
    

def main():
    np.random.seed(2024)

    NN = NeuralNetwork()
    NN.train()
    predictions = NN.predict()

    cost = lambda beta: ...




if __name__ == "__main__":
    main()