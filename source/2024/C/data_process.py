#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


# In[3]:


data = pd.read_csv(r"E:\code\meisai\source\2024\C\Wimbledon_featured_matches.csv", sep=',')
print(data.shape)
set_s=7 # H
game_s=9 #j
game_v=18 #S
set_v=19 #T
set_n=4 #E
game_n=5 #F
p_s=11 #L
server=13 #N
l=7283
x=np.zeros(7284)
num=0
global num
# for i in range(0,7284):
#     for j in range(0,51):
#         if data.iloc(i,j)== NaN:
#             print(i,j,data)
# 预计这里计算判空和数据处理
        


# In[4]:


def d1(i,p1,p2):
    if data.iloc[i,game_v]!= (p2+1):
        return
    if data.iloc[i,set_s+p1]-data.iloc[i,set_s+p2]==2:#如果当前point的set差为2，p1优势
        j=i
        while data.iloc[j,set_n]>i and data.iloc[j,set_n]<i+2 and j<=l:
            j=j+1
        if data.iloc[j,set_n]== i+2:#存在逆转
            if data.iloc[i,game_v]== (p2+1):#当前局是逆转者得分的则p设为转折点
                x[i]=1
    


# In[5]:


def d2(i,p1,p2):
    if data.iloc[i,game_v]!=p2+1:
        return
    j=i
    if data.iloc[i,set_n]==1:#第一局不用比
        return
    while data.iloc[j,set_n]==data.iloc[i,set_n] and j>0:
        j=j-1
    if data.iloc[j,game_s+p1]-data.iloc[j,game_s+p2]<3:#如果p1不是顺风比分差大于等于4则返回
        return
    k=i
    while data.iloc[k,set_n]==data.iloc[i,set_n] and k<=l:
        k=k+1
    if data.iloc[k,set_s+p2]<=data.iloc[i,set_s+p2]:
        return #如果p1顺风p2没有赢得胜利
    if data.iloc[i,game_v]==p2+1:
        x[i]=1


# In[6]:


def d3(i,p1,p2):
    if data.iloc[i,server]!=p1+1:#不是p1发球
        return
    if data.iloc[i,game_n]!=1:#不是set的第一个发球局
        return
    if data.iloc[i,game_v]!=p2+1:#不是p2胜利
        return
    x[i]=1


# In[7]:


def d4(i,p1,p2):
    if data.iloc[i,game_v]!=p2+1:#不是假定弱势赢则返回
        return
    if data.iloc[i-2,p_s+p1]!='AD':
        return
    x[i]=1


# In[8]:


def d5(i,p1,p2):
    if data.iloc[i,game_v]!=p2+1:#不是假定弱势赢则返回
        return
    if data.iloc[i,game_s+p1]-data.iloc[i,game_s+p2]<3:#不是当前set game差距3以上返回
        return
    j=i
    while data.iloc[j,set_n]==data.iloc[i,set_n] and j<=l:
        j=j+1#求下一个set
    if data.iloc[j,set_s+p2]<=data.iloc[i,set_s+p2]:#在劣势情况下没有胜利
        return
    x[i]=1


# In[9]:


for i in range(0,7283):
    d1(i,0,1)
    d1(i,1,0)
    d2(i,0,1)
    d2(i,1,0)
    d3(i,0,1)
    d3(i,1,0)
    d4(i,0,1)
    d4(i,1,0)
    d5(i,0,1)
    d5(i,1,0)
for i in range(0,l):
    if x[i]==1:
        num=num+1
print(num)
print(x)


# In[10]:


for j in range(42,45):
    for i in range(0,7283):
        if pd.isnull(data.iloc[i,j]):
            print(i,j)


# In[21]:


speed_mean=100.8045030203185
print(speed_mean)
for i in range(0,7283):
    if pd.isnull(data.iloc[i,42]):
        print(i)
        data.iloc[i,42]=speed_mean
df = pd.read_csv(r"E:\code\meisai\source\2024\C\Wimbledon_featured_matches.csv")
df['speed_mp2'] = data.iloc[:,42]
print(data.iloc[:,42])
df.to_csv('output.csv', index=False)


# In[22]:


serve_w=data.iloc[:,43]
a1=0
a2=0
a3=0
a4=0
a5=0
for i in range(0,7283):
    if not pd.isnull(serve_w.iloc[i]):
        if serve_w.iloc[i]=='B' :
            a1=a1+1
        if serve_w.iloc[i]=='BC':
            a2=a2+1
        if serve_w.iloc[i]=='BW':
            a3=a3+1
        if serve_w.iloc[i]=='C':
            a4=a4+1
        if serve_w.iloc[i]=='W':
            a5=a5+1
print(a1,a2,a3,a4,a5)
for i in range(0,7283):
    if pd.isnull(data.iloc[i,43]):
        data.iloc[i,43]='C'
df['speed_wide2'] = data.iloc[:,43]
df.to_csv('output.csv', index=False)


# In[23]:


a1=0
a2=0
for i in range(0,7283):
    if not pd.isnull(serve_w.iloc[i]):
        if data.iloc[i,44]=='CTL' :
            a1=a1+1
        if data.iloc[i,44]=='NCTL':
            a2=a2+1
print(a1,a2)
for i in range(0,7283):
    if pd.isnull(data.iloc[i,44]):
        data.iloc[i,44]='NCTL'
df['serve_depth2'] = data.iloc[:,44]
df.to_csv('output.csv', index=False)


# In[24]:


a1=0
a2=0
for i in range(0,7283):
    if not pd.isnull(serve_w.iloc[i]):
        if data.iloc[i,45]=='D' :
            a1=a1+1
        if data.iloc[i,45]=='ND':
            a2=a2+1
print(a1,a2)
for i in range(0,7283):
    if pd.isnull(data.iloc[i,45]):
        data.iloc[i,45]='ND'
df['return_depth2'] = data.iloc[:,45]
df.to_csv('output.csv', index=False)


# In[20]:


#放表里面
import numpy
import csv
x=x.reshape(-1,1)
x=pd.DataFrame(x)
df = pd.read_csv(r"E:\code\meisai\source\2024\C\Wimbledon_featured_matches.csv")
df['turnning_point'] = x

df.to_csv('output.csv', index=False)


# In[ ]:





# In[ ]:




