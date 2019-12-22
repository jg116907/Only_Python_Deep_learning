import numpy as np

# 지수 함수로 인해 너무 큰 값이 계산 -> Inf로 표시되면서 수치가 불안정해진다.
def softmax_overflow(a):
  exp_a = np.exp(a)
  sum_exp_a = sum(exp_a)
  return exp_a/sum_exp_a

# exp내의 배열을 배열 내 최대값을 뺴준 값으로 계산 -> overflow 방지
# def softmax(a): # 1d
#   c = np.max(a)
#   exp_a = np.exp(a-c)
#   sum_exp_a = sum(exp_a)
#   return exp_a/sum_exp_a

def softmax(x): # 2d
  if x.ndim == 2:
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 
  x = x - np.max(x) # 오버플로 대책
  return np.exp(x) / np.sum(np.exp(x))

## test
# x = np.arange(-10,10,1)
# y = softmax(x)
# print(y)
# print(np.sum(y))