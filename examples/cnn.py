import numpy as np
from .core_backend import CNN

class CNNClassifier:
    def __init__(self, lr=0.01, batch_size=32, epochs=10):
        self.model = CNN()
        self.model.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        X = X.reshape(len(X), -1) / 255.0
        for _ in range(self.epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), self.batch_size):
                b = idx[i:i+self.batch_size]
                xb = X[b].tolist()
                yb = y[b].tolist()
                self.model.train_batch(xb, yb)

    def predict(self, X):
        X = X.reshape(len(X), -1) / 255.0
        return np.array([self.model.predict(x.tolist()) for x in X])
