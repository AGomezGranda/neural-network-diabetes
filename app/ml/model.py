import numpy as np

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = {}
        self.L = len(self.layer_sizes) - 1
        self.n = 0
        self.costs = []
        self.initialize_parameters()
    
    def initialize_parameters(self):
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2 / self.layer_sizes[l-1])
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))