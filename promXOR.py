import numpy as np

def relu(x):
    return np.maximum(0, x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
bias = np.array([[0], [-1], [0]])
weight = np.array([[1, 1], [1, 1], [1, -2]])

for i in range(len(X)):
    h1 = X[i,0] * weight[0,0] + X[i,1] * weight[0,1] + bias[0]
    reluh1 = relu(h1)
    h2 = X[i,0] * weight[1,0] + X[i,1] * weight[1,1] + bias[1]
    reluh2 = relu(h2)
    y1 = reluh1 * weight[2,0] + reluh2 * weight[2,1] + bias[2]
    result = relu(y1)
    print("Result for", X[i, 0], X[i, 1], ":", result)