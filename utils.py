import numpy as np


#  Calculate the sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculate the derivative of the sigmoid function.
def sigmoid_derivative(x):
    return x * (1 - x)


def load_patterns(file_name):
    """
    Load training/testing patterns from a csv file.
    FORMAT EXAMPLE: "input1,input2,input3;output1,output2"
    :param file_name: The name of the csv file.
    :return: A tuple (inputs, outputs), where each element is a list of patterns.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

    inputs = []
    outputs = []
    for line in lines:
        input_str, output_str = line.strip().split(';')
        inputs.append(list(map(float, input_str.split(','))))
        outputs.append(list(map(float, output_str.split(','))))

    return inputs, outputs
