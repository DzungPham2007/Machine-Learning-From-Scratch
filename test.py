import numpy as np
import pandas as pd
z=np.array([[1,2,3,4,5,6,7,8],
           [1,2,3,4,5,6,7,8],
           [1,2,3,4,5,6,7,8],]
           )
b= np.exp(z) / np.sum(np.exp(z), axis = 0, keepdims=True) 

d = np.sum(-z * np.log(z), 0)

y=np.array([0,1,2,3,4,5,6,7,8,9])

y = np.eye(10)[y]
print(y)

h = np.argmax(z, axis = 0)
print(h.shape)

c = b.T
print(c[0])
