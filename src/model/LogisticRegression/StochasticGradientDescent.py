import numpy as np

class StochasticGradientDescent:
    def __init__(self, lr = 0.1, epochs = 10):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/ (1 + np.exp(-z));

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for idx in range(n_samples):
                xi = X_shuffled[idx]
                yi = y_shuffled[idx]

                linear_model = np.dot(xi, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_model)

                error = yi - y_predicted
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
    
    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(linear)
        return pred

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(linear)
        return np.where(pred >= 0.5, 1, 0)

    def compute_loss(self, X, y):
        n_samples = X.shape[0]
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss