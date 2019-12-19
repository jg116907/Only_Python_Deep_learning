import numpy as np

def cee(y,t):
  delta = 1e-7
  return -np.sum(t*np.log(y+delta))

def cee_batch_onehot(y,t): # one-hot encoding이 된 상태의 y
  if y.dim == 1:
    t = t.reshape(1,t.size)
    y = y.reshape(1,y.size)
  batch_size = y.shape[0]
  return -np.sum(t*np.log(y+1e-7))/batch_size

def cee_batch(y,t): # 정답 값으로 y가 들어올경우 
  if y.dim == 1:
    t = t.reshape(1,t.size)
    y = y.reshape(1,y.size)
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size


