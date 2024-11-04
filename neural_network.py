import itertools
from typing import Optional

import autograd.numpy as np # type: ignore


class NeuralNetwork:
    def __init__(
            self,
            network_input_size,
            layer_output_sizes,
            activation_funcs,
            activation_ders,
            cost_func,
            cost_der,
            optimizer,
            seed: Optional[int] = None
    ) -> None:
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_func = cost_func
        self.cost_der = cost_der
        self.optimizer = optimizer
        self.seed = seed

        self.layers = self.create_layers(network_input_size, layer_output_sizes)

    def create_layers(self, network_input_size: int, layer_output_sizes: list[int]) -> list:
        np.random.seed(self.seed)
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            std = np.sqrt(2 / (layer_output_size + i_size))
            W = np.random.normal(scale=std, size=(i_size, layer_output_size))
            # W = np.random.randn(layer_output_size, i_size).T * 0.1
            b = np.zeros(layer_output_size)
            # b = np.ones(layer_output_size) * 0.1
            layers.append((W, b))

            i_size = layer_output_size

        return layers

    @staticmethod
    def feed_forward(inputs: np.ndarray, layers: list, activation_funcs: list) -> np.ndarray:
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def feed_forward_saver(self, input):
        layer_inputs = []
        zs = []
        a = input
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    def backpropagation(self, input, target):
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
    
    def flatten_layers(self, layers):
        return list(itertools.chain(*layers))
    
    def reconstruct_layers(self, layers):
        flat_to_tuple = []
        for i in range(len(layers) // 2):
            flat_to_tuple.append((layers[2*i], layers[2*i+1])) 
        return flat_to_tuple
    
    def gradient(self, input, layers, target):
        layers = self.reconstruct_layers(layers)
        self.layers = layers
        layer_grads = self.backpropagation(input, target)
        layer_grads = self.flatten_layers(layer_grads)
        
        return layer_grads

    def train(self, input, target):
        self.optimizer.set_gradient(self.gradient)
        layers = self.flatten_layers(self.layers)
        layers = self.optimizer.gradient_descent(input, layers, target)
        self.layers = self.reconstruct_layers(layers)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.feed_forward(inputs, self.layers, self.activation_funcs)
    
    def cost(self, inputs, target):
        prediction = self.predict(inputs)
        return self.cost_func(prediction, target)
