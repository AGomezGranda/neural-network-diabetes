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

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward_propagation(self, X):
        self.A = {0: X}
        self.Z = {}
        for l in range(1, self.L + 1):
            self.Z[l] = np.dot(self.parameters[f"W{l}"], self.A[l-1]) + self.parameters[f"b{l}"]
            self.A[l] = self.sigmoid(self.Z[l])
        return self.A[self.L]
    
    def compute_cost(self, AL, Z):
        m = Z.shape[1]
        cost = -1/m * np.sum(Z * np.log(AL) + (1 - Z) * np.log(1 - AL))
        return np.squeeze(cost)
    
    def backward_propagation(self, X, Y):
        m = X.shape[1]
        grads = {}
        dA = - (np.divide(Y, self.A[self.L]) - np.divide(1 - Y, 1 - self.A[self.L]))
        for l in range(self.L, 0, -1):
            dZ = dA * self.A[l] * (1 - self.A[l])
            grads[f"dW{l}"] = 1/m * np.dot(dZ, self.A[l-1].T)
            grads[f"db{l}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.parameters[f"W{l}"].T, dZ)
        return grads
    
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    
