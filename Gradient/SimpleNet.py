import sys
import os
sys.path.insert(0,os.pardir + "/Activation_function")
sys.path.insert(0,os.pardir + "/Cost_function")
import numpy as np
from Cross_entropy_error import cee_batch
from Softmax import softmax
from Gradient import numerical_gradient

## 신경망 학습에서 기울기를 구하는 법
# 가중치 W와 손실 함수 L의 신경망이 있다.
# 이 경우 경사는 dL/dW로 나타낼 수 있다. => W값이 변화했을 때 L이 얼마나 변화하는가
class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2,3) # 정규분포로 초기화, dummy
  def predict(self, x):
    return np.dot(x,self.W)
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cee_batch(y, t)
    return loss

## Test
net = simpleNet()
print(net.W)
x = np.array([0.6,0.9])
p = net.predict(x)
print(p)
print(np.argmax(p)) # 최댓값의 인덱스
t = np.array([0,0,1]) # 정답 레이블
print(net.loss(x,t))

## gradient 값
# def f(W):
#   return net.loss(x,t)
f = lambda w: net.loss(x,t)
dW = numerical_gradient(f,net.W)
print(dW)


