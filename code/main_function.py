if __name__ == '__main__':

    HIDDEN_NODE_COUNT = 2
    EPOCHS = 100

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [0, 0],

    ])

    Y = np.array([
        [0],
        [1],
        [1],
        [0]])

    # Uruchamianie sieci neuronowej.
    loss_history, classification_error_history, weights1_history, weights2_history = run_neural_network(X,
                                                                                                        Y,
                                                                                                        HIDDEN_NODE_COUNT,
                                                                                                        EPOCHS)
