def run_neural_network(feature_data, label_data, hidden_node_count, epochs):
    mse = []
    classification_error_history = []
    weights1_history = []
    weights2_history = []

    nn = NeuralNetwork(feature_data, label_data, hidden_node_count)

    for epoch in range(epochs):
        nn.forward_propagation()
        mse.append(MSE(label_data, nn.output))
        nn.back_propagation()
        classification_error_history.append(classification_error(label_data, nn.output))
        weights1_history.append(nn.weights_input.copy())
        weights2_history.append(nn.weights_hidden.copy())

    return mse, classification_error_history, weights1_history, weights2_history