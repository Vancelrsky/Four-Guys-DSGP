import torch
import gzip
import csv
from torch.utils.data import DataLoader
import torch.optim 
import time
import math
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import  torch
import datetime
import  numpy as np
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
with gzip.open('cleaned_data.zip','rb') as file:
    feature_data = pd.read_csv(file,index_col=[0,1])

with gzip.open('new_label_data.zip','rb') as file:
    new_label_data = pd.read_csv(file,index_col=[0,1])

X_train, X_test, Y_train, Y_test = train_test_split(feature_data,new_label_data,test_size= 0.2, random_state = 6)
cell = tf.nn.RNNCellDeviceWrapper(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = cell.call(inputs, h0) #调用call函数

print(h1.shape) # (32, 128)

test_output = rnn(X_test[:15].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(Y_test[:10], 'real number')
