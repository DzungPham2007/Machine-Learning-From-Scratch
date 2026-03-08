import numpy as np

class MSE:
    def calculate(self, y_pred, y):
        return np.mean((y_pred - y)**2)

    def backward(self, y_pred, y):
        return (2 / y.shape[0]) * np.sum(y_pred - y)