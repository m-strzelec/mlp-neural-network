from utils import sigmoid, sigmoid_derivative
import numpy as np


class Neuron:
    """
    The Neuron class. This represents a neuron node in a neural network.

    :param num_inputs: The number of input connections.
    :param use_bias: Whether to use a bias.
    """
    def __init__(self, num_inputs, use_bias):
        self.use_bias = use_bias
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = np.random.uniform(-1, 1) if self.use_bias else 0.0
        self.output = 0
        self.inputs = None
        self.delta = 0
        self.prev_weight_updates = np.zeros(num_inputs)

    # Calculate the output of this neuron for the given inputs
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(np.dot(inputs, self.weights) + self.bias)
        # print(self.bias)
        return self.output

    # Update the weights of this neuron based on the error
    def update_weights(self, error, learning_rate, momentum_factor):
        self.delta = error * sigmoid_derivative(self.output)

        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.delta * self.inputs[i]\
                               + momentum_factor * self.prev_weight_updates[i]
            self.prev_weight_updates[i] = learning_rate * self.delta * self.inputs[i]

        if self.use_bias:
            self.bias += learning_rate * self.delta
