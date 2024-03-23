class NeuralNetwork:
    def __init__(self, features, labels, hidden_node_count):
        self.input = features
        self.weights_input = np.random.rand(self.input.shape[1], hidden_node_count)
        self.hidden = None
        self.weights_hidden = np.random.rand(hidden_node_count, 1)
        self.expected_output = labels
        self.output = np.zeros(self.expected_output.shape)

    def forward_propagation(self):
        hidden_weighted_sum = np.dot(self.input, self.weights_input)
        self.hidden = sigmoid(hidden_weighted_sum)
        output_weighted_sum = np.dot(self.hidden, self.weights_hidden)
        self.output = sigmoid(output_weighted_sum)

    def back_propagation(self):
        cost = self.expected_output - self.output
        weights_hidden_update = np.dot(self.hidden.T, (2 * cost * sigmoid_derivative(self.output)))
        weights_input_update = np.dot(self.input.T, (np.dot(2 * cost * sigmoid_derivative(self.output), self.weights_hidden.T) * sigmoid_derivative(self.hidden)))
        self.weights_hidden += weights_hidden_update
        self.weights_input += weights_input_update