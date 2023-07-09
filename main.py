import argparse
import matplotlib.pyplot as plt
from irises import mlp_irises
from mlp import MLP
from autoencoder import autoencoder


def process_args():
    # Description
    parser = argparse.ArgumentParser(description='Train and test a Multi-layer Perceptron (MLP) neural network.')

    # Required args
    parser.add_argument('-i', '--input-nodes', type=int, required=True, help='Number of input nodes in the MLP.')
    parser.add_argument('-hn', '--hidden-layer-neurons', type=int, required=True,
                        help='Number of neurons in the hidden layer.')
    parser.add_argument('-o', '--output-nodes', type=int, required=True, help='Number of output nodes in the MLP.')
    parser.add_argument('-b', '--bias', type=int, required=True, help='Whether to use bias nodes in the MLP.')
    parser.add_argument('-p', '--process', choices=['autoencoder', 'irises'], required=True,
                        help='Which dataset to process. Choices are "autoencoder" or "irises".')

    # Train args
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.5, help='Learning rate for MLP training.')
    parser.add_argument('--epochs', '--max-epochs', type=int, default=1000,
                        help='Maximum number of epochs for MLP training.')
    parser.add_argument('--err', '--desired-error', type=float, default=0.05, help='Desired error for MLP training.')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum factor for MLP training.')
    parser.add_argument('--record-interval', type=int, default=10,
                        help='Interval (in epochs) at which to record error during MLP training.')
    parser.add_argument('--train-file', type=str, default='train_log.txt',
                        help='File to save the error records during MLP training.')

    # Test args
    parser.add_argument('--record-items', type=str, default=None, help='A comma-separated list of items to record'
                        'during training. If None save all statistics. Choose from: input_pattern, total_error,'
                        'desired_output, output_errors, output_values, output_weights, hidden_values, hidden_weights')
    parser.add_argument('--test-file', type=str, default='test_log.txt',
                        help='File to save the statistics during MLP testing.')

    # Irises extra options
    parser.add_argument('--test-size', type=float, default=0.6,
                        help='Proportion of the iris data to use as a test set. Ignored if --process is "autoencoder".')
    parser.add_argument('--result-file', type=str, default='irises_results.txt',
                        help='File to save the results of testing the MLP on the iris data. '
                             'Ignored if --process is "autoencoder".')

    # Save/load network
    parser.add_argument('--save-network', action='store_true', help='Whether to save the trained MLP to a file.')
    parser.add_argument('--load-network', type=str, default=None,
                        help='If specified, load the MLP from this file instead of training a new one.')
    parser.add_argument('--network-file', type=str, default='mlp.pickle', help='File to save or load the trained MLP.')
    parser.add_argument('--model-file', action='store_true',
                        help='The file to save and load the trained model from for training sessions.')

    return parser.parse_args()


def plot_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    epochs = []
    errors = []
    for line in lines:
        split_line = line.split(',')
        epoch = int(split_line[0].split()[-1])
        error = float(split_line[1].split()[-1])
        epochs.append(epoch)
        errors.append(error)

    plt.plot(epochs, errors)
    plt.xlabel('Epoki')
    plt.ylabel('Błąd średniokwadratowy')
    plt.title(f'Błąd na przestrzeni epok - {args.process} lr: {args.lr} m: {args.momentum}')
    plt.grid(True)
    plt.show()
    plt.close()


if __name__ == "__main__":
    args = process_args()
    if args.model_file:
        network = MLP.load("mlp.pickle")
        print("Loading model from mlp.pickle.")
    elif args.load_network is not None:
        print(f"Loading model from {args.load_network}.")
        network = MLP.load(args.load_network)
    else:
        print(f"Creating a new model with {args.input_nodes} input nodes,"
              f" {args.hidden_layer_neurons} hidden nodes, and {args.output_nodes} output nodes.")
        network = MLP(args.input_nodes, [args.hidden_layer_neurons], args.output_nodes, args.bias)

    if args.process == 'autoencoder':
        autoencoder(network, args.lr, args.epochs, args.err, args.momentum, args.record_interval,
                    args.train_file, args.record_items, args.test_file)
    elif args.process == 'irises':
        mlp_irises(network, args.lr, args.epochs, args.err, args.momentum, args.record_interval,
                   args.train_file, args.record_items, args.test_file, args.test_size, args.result_file)

    print(f"Training and testing {args.process} finished.")

    if args.model_file:
        print("Saving model to mlp.pickle.")
        network.save("mlp.pickle")
    elif args.save_network:
        print(f"Saving model to {args.network_file}.")
        network.save(args.network_file)

    plot_data(args.train_file)
