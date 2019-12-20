import numpy as np
import sys, os
sys.path.insert(0,os.pardir + '/Activation_function')
from Sigmoid import sigmoid

X = np.array([3.5,2.2])
W1 = np.array([[0.2,0.4,0.6],[0.1,0.3,0.5]])
B1 = np.array([0.3,0.4,0.5])

A1 = np.dot(X,W1)+B1
Z1 = sigmoid(A1)

print("A1 : ",A1)
print("Z1 : ",Z1)

W2 = np.array([[0.8,0.9,1.4],[0.3,1.2,0.4]])
B2 = np.array([0.3,0.4])

A2 = np.dot(Z1,W2.T)+B2
Z2 = sigmoid(A2)

print("A2 : ",A2)
print("Z2 : ",Z2)

W3 = np.array([[0.2,1.3],[1.4,0.5]])
B3 = np.array([0.3,0.4])

A3 = np.dot(Z2,W3) + B3
Y = A3
### 출력층의 활성화 함수
# 회귀 : 항등 함수
# 2 클래스 분류 : sigmoid
# 다항 클래스 분류 : softmax
print("Y : ", Y)
