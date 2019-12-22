import sys,os
# sys.path.insert(0,os.pardir+"/Activation_function")
# sys.path.insert(0,os.pardir+"/Cost_function")
sys.path.insert(0,os.pardir+"/Simple_ANN")
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# 60000개의 훈련 데이터 중 임의로 100개의 데이터를 추려냄
# 100개의 미니 배치를 대상으로 SGD를 수행하여 매개변수를 갱신
(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# hyper parameter
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니 배치 크기
learning_rate = 0.1

# 1epoch 당 반복수
iter_per_epoch = max(train_size/batch_size,1)

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
for i in range(iters_num):
  # 미니 배치 획득
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # 기울기 계산
  # grad = network.numerical_gradient(x_batch,t_batch) # 순전파
  grad = network.gradient(x_batch,t_batch) # 성능 개선판 # 역전파

  # 매개 변수 계산
  for key in ('W1','b1','W2','b2'):
    network.params[key] -= learning_rate * grad[key]

  # 학습 경과 기록
  # loss = network.loss(x_batch, t_batch)
  # train_loss_list.append(loss)
  
  # if i%1000==0:
  #   print(x_batch.shape)
  #   print(t_batch.shape)
  #   print(network.loss(x_batch,t_batch))
  
  if i%iter_per_epoch==0:
    print("count : ",i)
    print("loss : ",network.loss(x_batch,t_batch))
    print("train_acc : ",network.accuracy(x_train,t_train))
    print("test_acc : ",network.accuracy(x_test,t_test))
    print("---------------------------------------------")