import numpy as np
import csv
from mlp import MLP
from sklearn.model_selection import train_test_split


# Load data from a CSV file and preprocess it. The last column is assumed to be the output label.
def load_and_preprocess_data(filename):
    # Open the file and read it line by line using the csv reader
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)

    # Convert the data to a NumPy array and convert all elements to floats
    data = np.array(data, dtype=float)

    # Split the data into input features (x) and labels (y)
    # The labels are assumed to be integer values in the last column
    x, y = data[:, :-1], data[:, -1].astype(int)

    # Normalize the feature data to the range [0, 1]
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    # One-hot encode the labels
    y_encoded = np.zeros((y.size, y.max() + 1))
    y_encoded[np.arange(y.size), y] = 1

    return x, y_encoded


# Split data into training and testing sets.
def split_data(x, y, test_size=0.3):

    # # Create an array of indices and shuffle it
    # indices = np.arange(x.shape[0])
    # np.random.shuffle(indices)
    #
    # # Determine the index at which to split the data
    # split = int(test_size * x.shape[0])
    # test_idx, train_idx = indices[:split], indices[split:]
    #
    # # Split the data
    # x_train, x_test = x[train_idx], x[test_idx]
    # y_train, y_test = y[train_idx], y[test_idx]

    # Split the data using train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test


def train_mlp(x_train, y_train, hidden_layer_sizes):
    mlp = MLP(num_inputs=x_train.shape[1], num_neurons_per_hidden_layer=hidden_layer_sizes,
              num_outputs=y_train.shape[1])
    mlp.train(x_train.tolist(), y_train.tolist())
    return mlp


# Evaluate a trained MLP model by calculating the confusion matrix, precision, recall, and F1 score
def evaluate_model(mlp, record_items, test_file, x_test, y_test):
    # Get the model's predictions for the test data
    y_pred, _ = mlp.test(x_test.tolist(), y_test.tolist(), record_items=record_items, record_file=test_file)

    # Convert the outputs and predictions to class labels
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate the confusion matrix
    cm = np.zeros((np.unique(y_test_classes).size, np.unique(y_test_classes).size), dtype=int)
    for a, p in zip(y_test_classes, y_pred_classes):
        cm[a][p] += 1

    # Calculate precision, recall, and F-measure
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * precision * recall / (precision + recall)

    # Prepare dictionary for per class metrics
    metrics_per_class = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Calculate total number of correctly classified instances
    total_correct = np.trace(cm)

    # Calculate number of correctly classified instances for each class
    correct_per_class = np.diag(cm)

    return cm, metrics_per_class, total_correct, correct_per_class


def mlp_irises(mlp, learning_rate, epochs, desired_error, momentum, record_interval,
               train_file, record_items, test_file, test_size, result_file):
    data_filename = "data.csv"
    result_filename = result_file

    x, y = load_and_preprocess_data(data_filename)
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=test_size)

    mlp.train(
        training_inputs=x_train.tolist(),
        training_outputs=y_train.tolist(),
        learning_rate=learning_rate,
        max_epochs=epochs,
        desired_error=desired_error,
        momentum_factor=momentum,
        error_record_interval=record_interval,
        error_record_file=train_file
    )

    cm, metrics_per_class, total_correct, correct_per_class = \
        evaluate_model(mlp, record_items, test_file, x_test, y_test)

    with open(result_filename, 'w') as f:
        f.write("Total number of correctly classified instances: " + str(total_correct))
        f.write("\nCorrectly classified instances [setosa versicolor virginica]: " + str(correct_per_class))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nPrecision per class:\n")
        for species, precision in zip(['setosa', 'versicolor', 'virginica'], metrics_per_class['precision']):
            f.write(f"{species}: {precision}\n")
        f.write("\nRecall per class:\n")
        for species, recall in zip(['setosa', 'versicolor', 'virginica'], metrics_per_class['recall']):
            f.write(f"{species}: {recall}\n")
        f.write("\nF1 Score per class:\n")
        for species, f1 in zip(['setosa', 'versicolor', 'virginica'], metrics_per_class['f1']):
            f.write(f"{species}: {f1}\n")

