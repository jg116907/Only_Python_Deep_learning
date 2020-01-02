import numpy as np

class Relu:
  def __init__(self):
    self.mask = None # instance 변수 # True/False로 구성된 np 배열
  def forward(self,x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx

## mask test
# x = np.array([[1.0,-0.5],[-2.0,3.0]])
# print(x)
# mask = (x<=0)
# print(mask)

# [[ 1.  -0.5]
#  [-2.   3. ]]  
# [[False  True] 
#  [ True False]]