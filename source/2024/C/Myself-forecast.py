#!/usr/bin/env python
# coding: utf-8

# In[471]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import transforms, datasets


# In[472]:


# 文件读取
def get_Data(data_path):
    data = pd.read_csv(r"E:\code\meisai\source\2024\C\tableConvert.com_bvqnxd.csv", sep=',')
    data.fillna(0, inplace=True)
    print(data)
    label=data.iloc[:,17] # 取最后一个特征作为标签
    data['elapsed_time'] = data['elapsed_time'].str.replace(':', '')
    #print(data['elapsed_time'])
    data=data.iloc[:,[13,18,13,42]]
    #data=data.iloc[:,['elapsed_time','consecutive_victories','p1_double_fault','p2_double_fault','p1_unf_err','p2_unf_err','distance_difference_p2_p1']]  # 以三个特征作为数据    
    print(data)
    #print(label.head())
    print(label)
    return data,label


# In[480]:


# 数据预处理
def normalization(data,label):

#     mm_x=MinMaxScaler() # 导入sklearn的预处理容器
#     mm_y=MinMaxScaler()
#     data=data.values    # 将pd的系列格式转换为np的数组格式
#     label=label.values
#     data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
#     label=label.reshape(-1, 1)
#     label=mm_y.fit_transform(label)
    train_data=data
    val_data=label
    train_data=train_data.select_dtypes(include='number')
    train_data_numpy = np.array(train_data)
    train_mean = train_data_numpy.mean()
    train_std  = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)

    val_data_numpy = np.array(val_data)
    val_data_numpy = (val_data_numpy - train_mean) / train_std
    val_data_tensor = torch.Tensor(val_data_numpy)

    

    return train_data_numpy,val_data_numpy


# In[481]:


# 时间向量转换
def split_windows(data,seq_length):
    data = np.array(data)
    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:(i+seq_length),:]
        _y=data[i+seq_length,-1]
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    print('x.shape,y.shape=\n',x.shape,y.shape)
    return x,y


# In[482]:


# 数据分离
def split_data(x,y,split_ratio):

    train_size=int(len(y)*split_ratio)
    test_size=len(y)-train_size

    x_data=Variable(torch.Tensor(np.array(x)))
    y_data=Variable(torch.Tensor(np.array(y)))

    x_train=Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train=Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test=Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test=Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test


# In[483]:


# 数据装入
def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):

    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    num_epochs=int(num_epochs)
    train_dataset=Data.TensorDataset(x_train,y_train)
    test_dataset=Data.TensorDataset(x_train,y_train)
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True) # 加载数据集,使数据集可迭代
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

    return train_loader,test_loader,num_epochs


# In[493]:


# 定义模型
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

# 定义一个类
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
        super(Net,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.fc=nn.Linear(hidden_size,output_size) # 全连接层

    def forward(self,x):
        # h_0=Variable(torch.zeros(self.num_layers,x.size(0),self.output_size))
        # c_0=Variable(torch.zeros(self.num_layers,x.size(0),self.output_size))# 初始化h_0和c_0

        # pred, (h_out, _) = self.lstm(x, (h_0, c_0))
        # h_out = h_out.view(-1, self.hidden_size)
        # out = self.fc(h_out)

        # e.g.  x(10,3,100) 三个句子，十个单词，一百维的向量,nn.LSTM(input_size=100,hidden_size=20,num_layers=4)
        # out.shape=(10,3,20) h/c.shape=(4,b,20)
        batch_size, seq_len = x.size()[0], x.size()[1]    # x.shape=(604,3,3)
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0)) # output(5, 30, 64)
        pred = self.fc(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


# In[494]:


# 参数设置
seq_length=6 # 时间步长
input_size=4
num_layers=6
hidden_size=12
batch_size=25
n_iters=5000
lr=0.001
output_size=1
split_ratio=0.9
path='.\tableConvert.com_bvqnxd.csv'
moudle=Net(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(moudle.parameters(),lr=lr)
print(moudle)


# In[495]:


# 数据导入
data,label=get_Data(path)
data,label=normalization(data,label)
x,y=split_windows(data,seq_length)
x=x.astype(float)
y=y.astype(float)
x_data,y_data,x_train,y_train,x_test,y_test=split_data(x,y,split_ratio)
train_loader,test_loader,num_epochs=data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size)


# In[496]:


# train
iter=0
for epochs in range(num_epochs):
  for i,(batch_x, batch_y) in enumerate (train_loader):
    outputs = moudle(batch_x)
    optimizer.zero_grad()   # 将每次传播时的梯度累积清除
    # print(outputs.shape, batch_y.shape)
    loss = criterion(outputs,batch_y) # 计算损失
    loss.backward() # 反向传播
    optimizer.step()
    iter+=1
    if iter % 100 == 0:
      print("iter: %d, loss: %1.5f" % (iter, loss.item()))


# In[497]:


print(x_data.shape)


# In[498]:


moudle.eval()
train_predict = moudle(x_data)


# In[499]:


def result(x_data, y_data):
  moudle.eval()
  train_predict = moudle(x_data)

  data_predict = train_predict.data.numpy()
  y_data_plot = y_data.data.numpy()
  y_data_plot = np.reshape(y_data_plot, (-1,1))  
  
#   data_predict = mm_y.inverse_transform(data_predict)
#   y_data_plot = mm_y.inverse_transform(y_data_plot)

  plt.plot(y_data_plot)
  plt.plot(data_predict)
  plt.legend(('real', 'predict'),fontsize='15')
  plt.show()

  print('MAE/RMSE')
  print(mean_absolute_error(y_data_plot, data_predict))
  print(np.sqrt(mean_squared_error(y_data_plot, data_predict) ))

result(x_data, y_data)
result(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




