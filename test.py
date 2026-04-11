import numpy as np
import pandas as pd
import cupy as cp
x = cp.arange(1,10001)
y = cp.arange(10000).reshape(1000,10)

print(cp.exp(x/10001))





