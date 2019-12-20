import numpy as np
from Gradient import numerical_gradient
# 수식
# x = x - lr*(df/dx)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x
  for i in range(step_num):
    grad = numerical_gradient(f,x)
    x -= lr*grad
  return x

## Test
# 경사 하강법으로 f(x0,x1) = x0^2 + x1^2 의 최솟값을 구하기
def function_2(x):
  return x[0]**2 + x[1]**2

init_x = np.array([-3.0,4.0])
res = gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100)
print(res)

# 학습률이 너무 클 때 => 너무 큰 값을 발산
init_x = np.array([-3.0,4.0])
res = gradient_descent(function_2,init_x=init_x,lr=10.0,step_num=100)
print(res)

# 학습률이 너무 작을 때 => 원래 값에서 거의 갱신되지 않는다
init_x = np.array([-3.0,4.0])
res = gradient_descent(function_2,init_x=init_x,lr=1e-10,step_num=100)
print(res)
