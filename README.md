# Multi-layer Perceptron (MLP) Neural Network

This project involves training and testing a Multi-layer Perceptron (MLP) Neural Network using two datasets: the iris dataset and an autoencoder dataset. The implemented MLP uses backpropagation for training, and supports various command-line parameters to customize the training and testing process.

## Usage

You can run the main script `main.py` from the command line and customize the behavior with several options.

Example run:
```shell
python main.py -i 4 -hn 2 -o 4 -b 1 -p irises --lr 0.6 --epochs 800 --momentum 0.5 --train-file irises_train_log.txt --test-file irises_test_log.txt --test-size 0.9 --save-network --network-file irises.pickle
```

## Parameters

- `-i`, `--input-nodes`: Number of input nodes in the MLP (Required).
- `-hn`, `--hidden-layer-neurons`: Number of neurons in the hidden layer (Required).
- `-o`, `--output-nodes`: Number of output nodes in the MLP (Required).
- `-b`, `--bias`: Whether to use bias nodes in the MLP (Required).
- `-p`, `--process`: Which dataset to process. Choices are "autoencoder" or "irises" (Required).

#### Training parameters:
- `--lr`, `--learning-rate`: Learning rate for MLP training (Default: 0.5).
- `--epochs`, `--max-epochs`: Maximum number of epochs for MLP training (Default: 1000).
- `--err`, `--desired-error`: Desired error for MLP training (Default: 0.05).
- `--momentum`: Momentum factor for MLP training (Default: 0.5).
- `--record-interval`: Interval (in epochs) at which to record error during MLP training (Default: 10).
- `--train-file`: File to save the error records during MLP training (Default: 'train_log.txt').

#### Test parameters:
- `--record-items`: A comma-separated list of items to record during training. If None save all statistics. Choose from: input_pattern, total_error, desired_output, output_errors, output_values, output_weights, hidden_values, hidden_weights (Default: None).
- `--test-file`: File to save the statistics during MLP testing (Default: 'test_log.txt').

#### Irises dataset specific options:
- `--test-size`: Proportion of the iris data to use as a test set. Ignored if `--process` is "autoencoder" (Default: 0.6).
- `--result-file`: File to save the results of testing the MLP on the iris data. Ignored if `--process` is "autoencoder" (Default: 'irises_results.txt').

#### Save/load network options:
- `--save-network`: Whether to save the trained MLP to a file (Default: False).
- `--load-network`: If specified, load the MLP from this file instead of training a new one (Default: None).
- `--network-file`: File to save or load the trained MLP (Default: 'mlp.pickle').
- `--model-file`: The file to save and load the trained model from for training sessions (Default: False).

## License
[MIT](https://choosealicense.com/licenses/mit/)
