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
        
        