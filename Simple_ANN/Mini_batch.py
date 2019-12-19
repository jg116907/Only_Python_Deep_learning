## 빅데이터를 학습할 경우 무작위의 데이터를 추출해서 학습 -> 미니 배치

import sys
import os
sys.path.insert(0,os.pardir + "/Activation_function")
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(flatten=True,normalize=False)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size) # 60000개의 데이터 중 10개를 랜덤하게 선택
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]