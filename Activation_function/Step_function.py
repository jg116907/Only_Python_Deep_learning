import numpy as np

def step_function(x):
  return np.array(x>0,dtype=np.int)

## test
# x = np.arange(-5.0,5.0,1)
# print(x)
# y = step_function(x)
# print(y)

