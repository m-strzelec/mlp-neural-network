from layer import Layer
import random
import numpy as np
import pickle


class MLP:
    """
    The MLP (Multi-Layer Perceptron) class. This represents a neural network.

    :param num_inputs: The number of input neurons.
    :param num_neurons_per_hidden_layer: A list where each element is the number of neurons in a hidden layer.
    :param num_outputs: The number of output neurons.
    :param use_bias: Whether to use a bias node.
    """
    def __init__(self, num_inputs, num_neurons_per_hidden_layer, num_outputs, use_bias):
        self.layers = []
        num_inputs_per_neuron = num_inputs

        for num_neurons in num_neurons_per_hidden_layer:
            self.layers.append(Layer(num_neurons, num_inputs_per_neuron, use_bias))
            num_inputs_per_neuron = num_neurons

        self.layers.append(Layer(num_outputs, num_inputs_per_neuron, use_bias))

    # Forward propagate the inputs through the network to generate an output
    def forward_propagate(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            curr_inputs = [neuron.calculate_output(curr_inputs) for neuron in layer.neurons]
        return curr_inputs

    # Backward propagate the error through the network to update the weights.
    def backward_propagate(self, expected_output, learning_rate, momentum_factor):
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                for j in range(len(self.layers[i].neurons)):
                    error = expected_output[j] - self.layers[i].neurons[j].output
                    self.layers[i].neurons[j].update_weights(error, learning_rate, momentum_factor)
            else:
                for j in range(len(self.layers[i].neurons)):
                    error = sum([neuron.weights[j] * neuron.delta for neuron in self.layers[i + 1].neurons])
                    self.layers[i].neurons[j].update_weights(error, learning_rate, momentum_factor)

    def train(self, training_inputs, training_outputs, learning_rate=0.5, max_epochs=1000,
              desired_error=0.05, momentum_factor=0.5, error_record_interval=10, error_record_file='training_log.txt'):
        with open(error_record_file, 'w') as file:
            training_data = list(zip(training_inputs, training_outputs))
            for epoch in range(max_epochs):
                random.shuffle(training_data)
                training_inputs, training_outputs = zip(*training_data)
                total_error = 0
                for inputs, expected_output in zip(training_inputs, training_outputs):
                    actual_output = self.forward_propagate(inputs)
                    self.backward_propagate(expected_output, learning_rate, momentum_factor)
                    total_error += sum([(expected - actual) ** 2 for expected, actual in
                                        zip(expected_output, actual_output)])
                total_error /= len(training_inputs)

                # Record the error every certain number of epochs
                if epoch % error_record_interval == 0:
                    file.write(f'Epoch: {epoch}, Global Error: {total_error}\n')

                if total_error <= desired_error:
                    break

    def test(self, testing_inputs, testing_outputs, record_items=None, record_file='test_log.txt'):
        outputs = []
        errors = []
        # if record_items is None, record all items
        if record_items is None:
            record_items = {'input_pattern', 'total_error', 'desired_output', 'output_errors',
                            'output_values', 'output_weights', 'hidden_values', 'hidden_weights'}
        else:
            record_items = set(record_items.split(','))

        with open(record_file, 'w') as file:
            for inputs, expected_output in zip(testing_inputs, testing_outputs):
                actual_output = self.forward_propagate(inputs)
                outputs.append(actual_output)
                error = [(expected - actual) ** 2 for expected, actual in zip(expected_output, actual_output)]
                errors.append(error)

                if 'input_pattern' in record_items:
                    file.write('Input pattern: ')
                    file.write(np.array2string(np.array(inputs), separator=', '))
                    file.write('\n')
                if 'total_error' in record_items:
                    file.write(f'Total error: {sum(error)}\n')
                if 'desired_output' in record_items:
                    file.write('Desired output: ')
                    file.write(np.array2string(np.array(expected_output), separator=', '))
                    file.write('\n')
                if 'output_errors' in record_items:
                    file.write('Output errors: ')
                    file.write(np.array2string(np.array(error), separator=', '))
                    file.write('\n')
                if 'output_values' in record_items:
                    file.write('Output values: ')
                    for neuron in self.layers[-1].neurons:
                        file.write(np.array2string(np.array(neuron.output), separator=', '))
                        file.write(', ')
                    file.write('\n')
                if 'output_weights' in record_items:
                    file.write('Output weights:\n')
                    for neuron in self.layers[-1].neurons:
                        file.write(np.array2string(np.array(neuron.weights), separator=', '))
                        file.write('\n')
                if 'hidden_values' in record_items:
                    file.write('Hidden values:')
                    for layer in self.layers[:-1]:
                        file.write('\n')
                        for neuron in layer.neurons:
                            file.write(np.array2string(np.array(neuron.output), separator=', '))
                            file.write(', ')
                    file.write('\n')
                if 'hidden_weights' in record_items:
                    file.write('Hidden weights:\n')
                    for layer in self.layers[:-1]:
                        for neuron in layer.neurons:
                            file.write(np.array2string(np.array(neuron.weights), separator=', '))
                            file.write('\n')

                file.write('\n')
        return outputs, errors

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
