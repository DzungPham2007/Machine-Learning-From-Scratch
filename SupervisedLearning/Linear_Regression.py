import numpy as np
import math
    
class linear_regression():
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random(-limit, limit, (n_features, ))
        self.b = 0

    def fit(self,x,y):
        #create weights
        self.initialize_weights(x.shape[1])
        n_samples = x.shape[0]

        #calculate weights using gradient descent
        for i in range(self.n_iterations):
            y_pred = x.dot(self.w) + self.b

            error = y_pred - y

            # Gradient for weights: (1/n) * X_transpose * error
            d_dw = (1/n_samples) * x.T.dot(error)
            d_db = 1/n_samples * np.sum(error)

            #update weights and bias
            self.w -= self.learning_rate*d_dw
            self.b -= self.learning_rate*d_db
    
    def predict(self, x):
        return x.dot(self.w) + self.b
            
