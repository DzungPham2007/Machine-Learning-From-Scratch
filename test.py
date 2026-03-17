import numpy as np
import pandas as pd
z=np.array([[1,2,3,4,5,6,7,8],
           [1,2,3,4,5,6,7,8],
           [1,2,3,4,5,6,7,8],]
           )
b= np.exp(z) / np.sum(np.exp(z), axis = 0, keepdims=True) 

d = np.sum(-z * np.log(z), 0)

print(f"z:{d}")

c = b.T
print(c[0])
