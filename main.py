from DeepLearning import NN
from LossFunction import MSE
from DeepLearning.NN import Dense, Sigmoid, SDG, ReLU, Softmax, CategoricalCrossEntropy
import numpy as np
import pandas as pd

train_data = pd.read_csv('./Datasets/Digit-Recognizer/train.csv')
print("Shape of train_data:", train_data.shape)

x = train_data.iloc[:, 1:]  
y = train_data.iloc[:, 0] 
print("Shape of X after separating features:", x.shape)
print("Shape of Y after separating features:", y.shape)

x = x / 255.0
x = np.array(x)

y = np.eye(10)[y]
print("Shape of y after one-hot encoding:", y.shape)

test_train_x = x[:30]
test_train_y = y[:30]
print(test_train_x.shape)

model = NN.Model(
    layers = [
        Dense(784, 128, activation = ReLU()),
        Dense(128, 64, activation = ReLU()),
        Dense(64, 10, activation = ReLU()),
        Dense(10, 10, activation = Softmax())
    ],
    loss_function = CategoricalCrossEntropy(),
    optimizer = SDG(learning_rate = 0.5)
)

model.fit(test_train_x.T,test_train_y.T, epochs = 10)