from core_math import dot_product, vector_addition, scalar_multiplication
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def compute_gradient(self, x, y):
        if y * (dot_product(x, self.weights) + self.bias) >= 1:
            dw = self.lambda_param * self.weights
            db = 0
        else:
            dw = self.lambda_param * self.weights - y * x
            db = -y
        return dw, db

    def train(self, data):
        n_samples, n_features = data.shape[0], data.shape[1] - 1
        self.initialize_parameters(n_features)
        
        for _ in range(self.n_iters):
            for idx, row in data.iterrows():
                x = row[:-1]
                y = row[-1]
                dw, db = self.compute_gradient(x, y)
                self.weights = vector_addition(self.weights, scalar_multiplication(-self.learning_rate, dw))
                self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.sign(dot_product(X, self.weights) + self.bias)
    
    def calculate_loss(self, X, y):
        distances = 1 - y * (dot_product(X, self.weights) + self.bias)
        distances[distances < 0] = 0  # max(0, distance)
        hinge_loss = self.lambda_param * np.dot(self.weights, self.weights) + np.mean(distances)
        return hinge_loss
