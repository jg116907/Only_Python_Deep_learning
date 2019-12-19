import numpy as np
import pickle
from dataset.mnist import load_mnist
# from PIL import Image
import sys
import os
sys.path.insert(0,os.pardir + "/Activation_function")
from Sigmoid import sigmoid
from Softmax import softmax
(x_train, t_train),(x_test, t_test) = load_mnist(flatten=True,normalize=False)

## image check
# print(x_test[0])
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

# def img_show(img):
#   pil_img = Image.fromarray(np.uint8(img))
#   pil_img.show()
  
# img = x_train[0]
# label = t_train[0]
# print(label)

# print(img.shape)
# img = img.reshape(28,28)
# print(img)

# img_show(img)

with open("sample_weight.pkl","rb") as f:
  network = pickle.load(f)
  
def predict(network,x):
  W1,W2,W3 = network['W1'],network['W2'],network['W3']
  b1,b2,b3 = network['b1'],network['b2'],network['b3']


  a1 = np.dot(x,W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1,W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2,W3) + b3
  y = softmax(a3)

  return y

batch_size = 100
acc_cnt = 0
for i in range(0,len(x_test),batch_size):
  x_batch = x_test[i:i+batch_size]
  y_batch = predict(network,x_batch)
  p = np.argmax(y_batch,axis=1)
  acc_cnt += np.sum(p==t_test[i:i+batch_size])
  
print("accuracy : ",float(acc_cnt)/len(x_test))