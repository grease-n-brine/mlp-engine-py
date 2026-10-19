import numpy as np

class MLP:
    def __init__(self, layer, learning_rate=0.01):
        self.layer = layer
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layer) - 1):
            self.weights.append(np.random.randn(layer[i], layer[i + 1]))
            self.biases.append(np.random.randn(layer[i + 1]))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, x):
        self.activations = [x]
        self.zs = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)

            if i == len(self.weights) - 1:
                activation = z  # No activation function for the output layer
            else:
                activation = self._relu(z)
                
            self.activations.append(activation)
        