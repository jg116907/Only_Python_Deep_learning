import sys, os
sys.path.insert(0,os.pardir+"/Activation_function")
sys.path.insert(0,os.pardir+"/Cost_function")
import numpy as np
from Gradient import numerical_gradient
from Sigmoid import sigmoid
from Softmax import softmax
from Cross_entropy_error import cee_batch

# 확률적으로 무작위로 골라낸 데이터에 대해 경사 하강법을 수행 => SGD
class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 가중치 초기화
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    y = softmax(a2)
    return y
  def loss(self, x, t):
    y = self.predict(x)
    return cee_batch(y,t)
  def accuracy(self,x,t):
    y = self.predict(x)
    y = np.argmax(y,axis=1)
    t = np.argmax(t,axis=1)
    accuracy = np.sum(y==t) / float(x.shape[0])
    return accuracy
  def numerical_gradient(self,x,t): # 순전파
    loss_W = lambda w: self.loss(x,t)
    grads = {}
    grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
    return grads

  def sigmoid_grad(self, x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
  def gradient(self, x, t): # 역전파
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    grads = {}
    
    batch_num = x.shape[0] # 미니 배치로 들어오는 훈련 데이터의 개수 # (100, 784) 중 100
    
    # forward
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    
    # backward -> 
    dy = (y - t) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)
    
    da1 = np.dot(dy, W2.T)
    dz1 = self.sigmoid_grad(a1) * da1
    grads['W1'] = np.dot(x.T, dz1)
    grads['b1'] = np.sum(dz1, axis=0)
    return grads

## Test
# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

# x = np.random.randn(100,784) # 100개 분량의 입력 데이터
# y = net.predict(x)
# t = np.random.randn(100,10) # 100개 분량의 레이블 데이터

# loss_W = lambda w: net.loss(x,t)
# print(numerical_gradient(loss_W,net.params['W1']))

# 계산 시간이 매우 오래 걸림
# grads = net.numerical_gradient(x,t) # 기울기 계산
# print(grads['W1'].shape)
# print(grads['b1'].shape)
# print(grads['W2'].shape)
# print(grads['b2'].shape)
