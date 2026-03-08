import numpy as np
import math
import pickle

class ReLU():
    def forward(self, z):
        return np.maximum(0, z)
    
    #derivative of ReLU is 0(x<0) and 1(x>0)
    def backward(self, z):
        return (z>0).astype(float)
    
class Sigmoid():
    def forward(self, z):
        return 1 / (1 + np.exp(-z))
    
    def backward(self, z):
        s = self.forward(z)
        return s * (1-s)
    
class Softmax():
    def forward(self, z):
        # Minus z with max value of z to avoid overflow of e^z
        # z.shape = (num_class, batch)
        # output.shape = (num_class, batch)
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis = 1, keepdims=True) 

class BinaryCrossEntropy:
    def forward(self, y_pred, y):
        #y_pred.shape    = (1, batch)
        #y.shape         = (1, batch)  (0 or 1)
        #output.shape    = (1, batch)
        return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

class CategoricalCrossEntropy:
    def forward(self, y_pred, y):
        #y_pred.shape    = (num_class, batch)
        #y.shape         = (num_class, batch)
        #output.shap     = (1, batch) - num_class has been sum
        return np.sum(-y * np.log(y_pred), 0)

    def backward(self, z):
        return -1/z

class Flatten:
    def __init__(self, input_shape):
        self.x = input_shape[0]
        self.y = input_shape[1]
        self.batch = input_shape[2]

    def flatten(self, x):
        # Flatten 3D (batch, m, n) inputs -> 2D (batch, m)
        return x.reshape(self.batch,-1)

class Dense:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01   #Create 2D-weights from gaussian disstricution (n_inputs x n_neurons)
        self.d_weights = None
        self.bias = np.zeros((1,n_neurons))   #Create array bias lenth
        self.d_bias = None
        self.activation = activation
        self.z = None
        self.inputs = None
        self.outputs = None

    def forward(self,inputs):
        if isinstance(self.activation, Softmax):
            self.outputs = self.activation.forward(inputs)
        else:
            # inputs.shape  = inputs, batch
            # weights.shape = inputs, neurons
            # bias.shape    = 1, neurons
            # output.shape  = neurons, batch
            self.inputs = inputs
            self.z = np.dot(self.weights.T, inputs) + self.bias.T
            self.outputs = self.activation.forward(self.z)

        return self.outputs
    
    def backward(self, d_values):
        """
        C: Cost
        L: loss
        a(L) = f(z(L))
        z(L) = w(L) * a(L-1) + b(L)

        delta   = dL/dz(L) = dL/dz(L+1) * f'(L)     |   (n_(L), m)*(n_(L), m) = (n_(L), m)
        dC/dW   = (1/m) * w(L+1) * dz(L+1) * f'(L)  |   
                = (1/m) * d_input_pre * w(L-1)T     |   (n_(L), n_(L-1))
                = (1/m) * d_input_pre * w(L-1)T 
        dC/dB   = (1/m) * sum(delta)                |   (n_(L), 1)
        d_input = w(L) * dL/dz(L)
        """

        if not isinstance(self.activation, Softmax):
            #num of batch
            m = d_values.shape[1]

            #delta = dL/dz(L) = dL/dz(L+1) * f'(L) (n_i,m)*(n_i,m) = (n_i,m)
            self.delta =  d_values * self.activation.backward(self.z)

            #dC/dw = dz/dw * da/dz * dc/da
            #dC/dw = a * delta
            self.d_weights = (1/m) * np.dot(self.delta, self.inputs.T)

            #dC/db = dz/db * da/dz * dc/da
            #dC/dw = delta
            self.d_bias = (1/m) * np.sum(self.delta)
        
            #w(L)T * dL/dz(L)
            self.d_input = np.dot(self.weights, d_values)

            print(f"weights: {self.weights.shape} bias:{self.bias.shape}")
            print(f"d_weights: {self.d_weights.shape} d_bias:{self.d_bias.shape}")
            return self.d_input
    
class SDG:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
    def update(self, layers):
        for layer in layers:
            layer.weights -= layer.d_weights * self.learning_rate
            layer.bias -= layer.d_bias * self.learning_rate


class Model:
    def __init__(self, layers=None, loss_function=None, optimizer=None):
        self.layers = layers
        self.loss = loss_function
        self.optimizer = optimizer
        self.loss_value = None

    def fit(self,x,y,epochs):
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            # 1. FORWARD PASS
            output = x
            for layer in self.layers:
                output = layer.forward(output)
                print(output.shape)

            # 2. CALCULATE LOSS
            #recent layer is final layer
            if isinstance(self.loss, CategoricalCrossEntropy) and isinstance(self.layers[-1].activation, Softmax):
                #Softmax combine with CrossEntropy: dC/da_(L-1)(i) = a_(L)(i) - y_i (value a_(L) is softmax layer, a_(L-1) is previous layer)
                self.loss_value = output - y
                print("yes")
            else:
                print("no")

            # 3. BACKWARD PASS
            # Start the chain with the loss gradient
            d_values = self.loss_value

            print(f"d_values:{d_values.shape}")

            for layer in reversed(self.layers[:-1]):
                d_values = layer.backward(d_values)
                print(f"d_values:{d_values.shape}")

            # 4. UPDATE WEIGHTS AND BIAS
            #self.optimizer.update(self.layers)
            for layer in self.layers[:-1]:
                d_weights = layer.d_weights
                d_bias = layer.bias
                print(f"d_weights:{d_weights.shape} d_bias:{d_bias.shape}")
                layer.weights -= d_weights.T * 0.1
                layer.bias -= d_bias * 0.1

    def save(self, filename):
        model_data = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                model_data.append({
                    'weights' : layer.weights ,
                    'bias' : layer.bias 
                })

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'wb') as f:
            model_data = pickle.load(f)

        data_index = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = model_data[data_index]['weights']
                layer.bias = model_data[data_index]['bias']
                data_index += 1
        
        print(f"Model loaded from {filename}")
        

    def predict(self,x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        
        return output

        




            

            




