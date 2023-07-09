from mlp import MLP


training_input = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
training_output = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]


def autoencoder(network, learning_rate, epochs, desired_error, momentum,
                record_interval, train_file, record_items, test_file):

    network.train(
        training_inputs=training_input,
        training_outputs=training_output,
        learning_rate=learning_rate,
        max_epochs=epochs,
        desired_error=desired_error,
        momentum_factor=momentum,
        error_record_interval=record_interval,
        error_record_file=train_file
    )

    network.test(
        testing_inputs=training_input,
        testing_outputs=training_output,
        record_items=record_items,
        record_file=test_file
    )


# # Część badawcza autoenkoder
# autoenkoder = MLP(4, [2], 4, use_bias=True)
#
# autoenkoder.train(training_input, training_output, learning_rate=0.6, momentum_factor=0.0)
#
# hidden_layer_outputs = [neuron.output for neuron in autoenkoder.layers[1].neurons]
# print("Hidden layer outputs:", hidden_layer_outputs)
#
# lr_momentum_combinations = [(0.9, 0.0), (0.6, 0.0), (0.2, 0.0), (0.9, 0.6), (0.2, 0.9)]
#
# for lr, m in lr_momentum_combinations:
#     autoenkoder = MLP(4, [2], 4, use_bias=True)
#     autoenkoder.train(training_input, training_output, learning_rate=lr, max_epochs=1000, momentum_factor=m,
#                       error_record_file=f'autoencoder_training_log_lr_{lr}_momentum_{m}.txt')
#
#     predicted, _ = autoenkoder.test(training_input, training_output, record_file='autoencoder_test_log.txt')
#     print(f'Predicted: {predicted}')
