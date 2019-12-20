import numpy as np

# x0,x1 두 가지 변수가 있는 함수의 편미분을 동시에 계산
# (df/dx0, df/dx1)과 같이 모든 변수의 편미분을 벡터로 정리 => 기울기
def numerical_gradient_1d(f,x):
  h = 1e-4
  grad = np.zeros_like(x) # x와 형태가 같은 배열

  for idx in range(x.size):
    tmp_val = x[idx]
    # f(x+h) 계산
    x[idx] = tmp_val + h
    fxh1 = f(x)
    # f(x-h) 계산
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2)/(2*h)
    x[idx] = tmp_val
  return grad

def numerical_gradient(f,x):
  h = 1e-4
  grad = np.zeros_like(x) # x와 형태가 같은 배열

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = float(tmp_val) + h
    fxh1 = f(x) # f(x+h)
    
    x[idx] = tmp_val - h 
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)
    
    x[idx] = tmp_val # 값 복원
    it.iternext()
  return grad

## Test
# def function_2(x):
#   return x[0]**2 + x[1]**2 
# print(numerical_gradient(function_2,np.array([3.0,4.0])))
# print(numerical_gradient(function_2,np.array([0.0,2.0])))
# print(numerical_gradient(function_2,np.array([3.0,0.0])))

# 해당 기울기 들이 가리키는 곳은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향
