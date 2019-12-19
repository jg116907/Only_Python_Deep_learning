import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(-x))

## test
# x = np.arange(-50,50,1)
# y = sigmoid(x)
# print(y)
