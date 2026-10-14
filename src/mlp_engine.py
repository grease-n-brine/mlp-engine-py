import numpy as np

class MLP:
    def __init__(self, layer, learning_rate=0.01):
        self.layer = layer
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        