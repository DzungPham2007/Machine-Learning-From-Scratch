from DeepLearning import NN
from DeepLearning.NN import Dense, SGD, ReLU, Softmax, CategoricalCrossEntropy, Conv2D, MaxPooling, Flatten
import numpy as np
import cupy as cp
import pandas as pd

train_data = pd.read_csv('./Datasets/Digit-Recognizer/train.csv')
print("Shape of train_data:", train_data.shape)

x = train_data.iloc[:, 1:]  
y = train_data.iloc[:, 0] 
print("Shape of X after separating features:", x.shape)
print("Shape of Y after separating features:", y.shape)

x = x / 255.0
x = cp.array(x)
x = x.reshape(42000,1,28,28)

x_train = x[:30000]
x_valid = x[30000:]
print("Shape of x:", x_train.shape, x_valid.shape)

y_cpu = np.eye(10)[y]
y_gpu = cp.asarray(y_cpu)

y_train = y_gpu[:30000]
y_valid = y_gpu[30000:]
print("Shape of y after one-hot encoding:", y_train.shape, y_valid.shape)
print(y_train[:5])

BATCH_SIZE = 1024

model = NN.Model(
    layers = [
        Conv2D(num_kernel = 32, kernel_size = (5, 5), input_shape = (BATCH_SIZE, 1, 28, 28), activation = ReLU()),
        MaxPooling(kernel_size = (2, 2), stride = 2),
        Conv2D(num_kernel = 16, kernel_size = (5, 5), input_shape = (BATCH_SIZE, 32, 12, 12), activation = ReLU()),
        MaxPooling(kernel_size = (2, 2), stride = 2),
        Flatten(),
        Dense(256, 128, activation = ReLU()),
        Dense(128, 10, activation = ReLU()),
        Dense(10, 10, activation = Softmax())
    ],
    loss_function = CategoricalCrossEntropy(),
    optimizer = SGD()
)
print(x_train.shape)

loss_value = model.fit(x_train, y_train, BATCH_SIZE, epochs = 10, x_validation = x_valid, y_validation = y_valid)

print(loss_value)
xpoints = np.arange(0,x.shape[0] // BATCH_SIZE)
ypoints = np.zeros(x.shape[0] // BATCH_SIZE)


