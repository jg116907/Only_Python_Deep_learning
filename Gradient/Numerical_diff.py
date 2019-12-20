import numpy as np

## Issue 1.
# 수치 미분 공식을 그대로 구현 -> h의 값을 최대한 0에 가깝게
# 이 방식은 rounding error를 발생시킨다.
# eg) np.float32(1e-50) => 0.0
def numerical_diff(f, x): 
  h = 1e-4 # 적당히 10^-4 를 사용
  return (f(x+h) - f(x))/h

## Issue 2.
# x위치의 함수의 기울기를 구한는 것이 미분의 목적
# 하지만 위 공식은 x+h와 x 사이의 함수의 기울기를 구하는 공식이다. => 오차가 존재
# 이 오차를 줄이기 위해 x+h 와 x-h의 차분을 이용 => 중심 차분
def numerical_diff_center(f, x):
  h = 1e-4 # 0.0001
  return (f(x+h) - f(x-h))/(2*h)


## Test
# 해석적 해는 df(x)/dx = 0.02x + 0.1
# x=5 => 0.2
# x=10 => 0.3

def function_1(x):
  return (0.01*x**2) + (0.1*x)

x = np.arange(0.0,20.0,0.1)
y = function_1(x)

print(numerical_diff_center(function_1,5)) # x값이 5일 때 function_1 의 기울기(변화량)
print(numerical_diff_center(function_1,10))
print(numerical_diff_center(function_1,x))

## 편미분
def function_2(x): # 변수가 두 개
  return x[0]**2 + x[1]**2 

# x0=3, x1=4 일 때 x0에 대한 편미분 df/dx0
def function_tmp1(x0):
  return x0*x0 + 4.0**2.0
print(numerical_diff_center(function_tmp1,3.0))

# x0=3, x1=4 일 때 x1에 대한 편미분 df/dx1
def function_tmp2(x1):
  return 3.0**2.0 + x1*x1 
print(numerical_diff_center(function_tmp2,4.0))

