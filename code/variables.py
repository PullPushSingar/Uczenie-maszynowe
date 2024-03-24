

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

HIDDEN_NODE_COUNT = 2
EPOCHS = 100
mse = []
classification_error_history = []
weights1_history = []
weights2_history = []