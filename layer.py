from neuron import Neuron


class Layer:
    """
    The Layer class. This represents a layer of neurons in a neural network.

    :param num_neurons: The number of neurons in this layer.
    :param num_inputs_per_neuron: The number of input connections per neuron.
    :param use_bias: Whether to use a bias node.
    """
    def __init__(self, num_neurons, num_inputs_per_neuron, use_bias):
        self.neurons = [Neuron(num_inputs_per_neuron, use_bias) for _ in range(num_neurons)]
