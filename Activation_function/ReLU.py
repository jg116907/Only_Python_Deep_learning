# Recitified Linear Unit
# 입력이 0을 넘으면 그 입력을 그대로 출력, 0 이하면 0을 출력하는 함수
import numpy as np

def relu(x):
  return np.maximum(0,x)

## test
# x = np.arange(-10,20,1)
# y = relu(x)
# print(y)