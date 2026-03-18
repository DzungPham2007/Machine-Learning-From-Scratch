from DeepLearning import NN
from DeepLearning.NN import Dense, SGD, ReLU, Softmax, CategoricalCrossEntropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./Datasets/Digit-Recognizer/train.csv')
print("Shape of train_data:", train_data.shape)

x = train_data.iloc[:, 1:]  
y = train_data.iloc[:, 0] 
print("Shape of X after separating features:", x.shape)
print("Shape of Y after separating features:", y.shape)

x = x / 255.0
x = np.array(x)

x_train = x[:30000]
x_valid = x[30000:]
print("Shape of x:", x_train.shape, x_valid.shape)

y = np.eye(10)[y]

y_train = y[:30000]
y_valid = y[30000:]
print("Shape of y after one-hot encoding:", y_train.shape, y_valid.shape)
print(y_train[:5])

model = NN.Model(
    layers = [
        Dense(784, 128, activation = ReLU()),
        Dense(128, 64, activation = ReLU()),
        Dense(64, 10, activation = ReLU()),
        Dense(10, 10, activation = Softmax())
    ],
    loss_function = CategoricalCrossEntropy(),
    optimizer = SGD()
)

BATCH_SIZE = 128
loss_value = model.fit(x_train, y_train, BATCH_SIZE, epochs = 10, x_validation = x_valid, y_validation = y_valid)

print(loss_value)
xpoints = np.arange(0,x.shape[0] // BATCH_SIZE)
ypoints = np.zeros(x.shape[0] // BATCH_SIZE)


