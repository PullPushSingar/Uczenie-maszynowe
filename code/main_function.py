if __name__ == '__main__':
    FEATURE_COUNT = 2
    HIDDEN_NODE_COUNT = 2
    EPOCHS = 100

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    run_neural_network(X, Y, FEATURE_COUNT, HIDDEN_NODE_COUNT, EPOCHS)
