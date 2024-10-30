import itertools

import autograd.numpy as np # type: ignore
from sklearn import datasets # type: ignore
from sklearn.metrics import accuracy_score # type: ignore


class NeuralNetwork:
    def __init__(
            self,
            network_input_size,
            layer_output_sizes,
            activation_funcs,
            activation_grads,
            cost_func,
            cost_grad,
            optimizer,
    ) -> None:
        self.activation_funcs = activation_funcs
        self.activation_grads = activation_grads
        self.cost_func = cost_func
        self.cost_grad = cost_grad
        self.optimizer = optimizer

        self.layers = self._create_layers(network_input_size, layer_output_sizes)

    def _create_layers(self, network_input_size: int, layer_output_sizes: list[int]) -> list:
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size).T
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size

        return layers

    @staticmethod
    def _feed_forward(inputs: np.ndarray, layers: list, activation_funcs: list) -> np.ndarray:
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def _feed_forward_saver(self, input):
        layer_inputs = []
        zs = []
        a = input
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a
    
    def _backpropagation(self, input, target):
        layer_inputs, zs, predict = self._feed_forward_saver(input)
        num_layers = len(self.layers)
        layer_grads = [None] * num_layers

        for i in reversed(range(num_layers)):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_grads[i]

            if i == num_layers - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_grad(predict, target)
                # print(f"dC_da, first iter: {dC_da}")
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da * activation_der(z)
            # print(f"dC_dz: {dC_dz}")
            dC_dW = layer_input.T @ dC_dz / len(self.layers[-1][1])
            # print(f"dC_dW")
            dC_db = np.mean(dC_dz, axis=0) / len(self.layers[-1][1])*len(layer_input) # deriv wrt b is 1
            # print(f"dC_db: {dC_db}")

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

        # layer_grads = [() for layer in self.layers]
        # num_layers = len(self.layers)

        # for i in reversed(range(num_layers)):
        #     layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_grads[i]

        #     if i == num_layers - 1:
        #         dC_da = self.cost_grad(predict, target)
        #     else:
        #         (W, b) = self.layers[i + 1]
        #         dC_da = dC_dz @ W

        #     dC_dz = dC_da @ activation_der(z)
        #     dC_dW = dC_dz @ np.tensordot(np.eye(len(z)), layer_input, axes=0)
        #     dC_db = dC_dz

        #     layer_grads[i] = (dC_dW, dC_db)
        
        # return layer_grads
    
    def gradient(self, input, layers, target):
        flat_to_tuple = []
        for i in range(len(layers) // 2):
            flat_to_tuple.append((layers[2*i], layers[2*i+1]))
        layers = flat_to_tuple
        self.layers = layers
        layer_grads = self._backpropagation(input, target)
        layer_grads = list(itertools.chain(*layer_grads))
        return layer_grads

    def train(self, input, target):
        self.optimizer.set_gradient(self.gradient)
        layers = self.layers
        layers = list(itertools.chain(*layers))
        self.layers = self.optimizer.gradient_descent(input, layers, target)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        flat_to_tuple = []
        for i in range(len(self.layers) // 2):
            flat_to_tuple.append((self.layers[2*i], self.layers[2*i+1]))
        layers = flat_to_tuple
        return self._feed_forward(inputs, layers, self.activation_funcs)
