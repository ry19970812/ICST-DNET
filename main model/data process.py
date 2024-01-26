import torch.nn.functional as F
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch.utils.data as Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import copy
import random




# 使用cuda 选择GPU
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# print(device)
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #指定GPU编号
device = torch.device("cuda") #创建单GPU对象


all_data = np.load('......')


# 生成训练集

float_data = all_data[0:6832, :]
# print(float_data.shape) # shape (6832, 108)

# 训练集归一化

float_data = np.reshape(float_data, (6832 * 108, -1))
# print(float_data.shape)
float_data = float_data.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1)) #feature_range表示设置归一化范围为（0,1）
float_data = scaler.fit_transform(float_data)  # 将float_data按列转换为（0,1）之间

float_data = np.reshape(float_data, (6832, 108))

float_data = torch.FloatTensor(float_data)

# print(float_data.shape) # shape torch.Size([6832, 108])


# print(float_data[0])

# the input of ST
X_train_input_ST = []

# closeness period
X_train_enc_input = []
# X_train_dec_input = []
#
# X_train_dec_output = []
X_train_output = []
# # day period
X_train_input_day = []

# # week period
X_train_input_week = []



# SE

SE_initial = np.load('./Ningxia-YC/SE_all_Ningxia-YC.npy')
# TE = np.load('TE.npy')
# TE = torch.FloatTensor(TE)
# print(SE.shape) # shape (29, 64)
# print(TE.shape) # shape (17000, 295)

SE_train_input = torch.FloatTensor(SE_initial)

SE_train_input = torch.unsqueeze(SE_train_input, dim=0)

# TE

TE_train_input = torch.zeros([4804, 17*2])
TE_train_input = torch.FloatTensor(TE_train_input)



for i in range(2016, 6820):

    X_train_input_ST.append(float_data[i-12:i])

    X_train_enc_input.append(float_data[i-12:i])
    X_train_output.append(float_data[i:i+12])






X_train_enc_input = torch.stack(X_train_enc_input)
X_train_output = torch.stack(X_train_output)



# the input of ST
# ------------------------------------------------
X_train_input_ST = torch.stack(X_train_input_ST)

# ------------------------------------------------


# the input of casual diffusion
# ------------------------------------------------
X_train_input_CD = X_train_enc_input

# ------------------------------------------------







# 生成交叉验证集

validation_float_data = all_data[0:7832, :]


# 交叉验证集归一化

validation_float_data = np.reshape(validation_float_data, (7832 * 108, -1))
# print(float_data.shape)
validation_float_data = validation_float_data.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1)) #feature_range表示设置归一化范围为（0,1）
validation_float_data = scaler.fit_transform(validation_float_data)  # 将float_data按列转换为（0,1）之间

validation_float_data = np.reshape(validation_float_data, (7832, 108))

validation_float_data = torch.FloatTensor(validation_float_data)


# print(validation_float_data.shape) # shape (1000, 108) (time_series, num_sensors)


# the input of ST
X_validation_input_ST = []


#
# closeness period
X_validation_enc_input = []
# X_validation_dec_input = []
#
# X_validation_dec_output = []
X_validation_output = []

# # day period
X_validation_input_day = []
#
# # week period
X_validation_input_week = []

# SE

SE_initial = np.load('.......')
# TE = np.load('TE.npy')
# TE = torch.FloatTensor(TE)
# print(SE.shape) # shape (29, 64)
# print(TE.shape) # shape (17000, 295)

SE_validation_input = torch.FloatTensor(SE_initial)

SE_validation_input = torch.unsqueeze(SE_validation_input, dim=0)

# TE

TE_validation_input = torch.zeros([988, 17*2])
TE_validation_input = torch.FloatTensor(TE_validation_input)


for i in range(6832, 7820):
    X_validation_enc_input.append(validation_float_data[i - 12:i])
    X_validation_output.append(validation_float_data[i:i+12])
    X_validation_input_ST.append(validation_float_data[i-12:i])


X_validation_enc_input = torch.stack(X_validation_enc_input)
X_validation_output = torch.stack(X_validation_output)


# the input of ST
# ------------------------------------------------
X_validation_input_ST = torch.stack(X_validation_input_ST)

# ------------------------------------------------


# the input of casual diffusion
# ------------------------------------------------
X_validation_input_CD = X_validation_enc_input

# ------------------------------------------------



# # 生成测试集
test_float_data = all_data[0:8832, :]

# 测试集归一化

test_float_data = np.reshape(test_float_data, (8832 * 108, -1))


# 计算sigma and mu
test_float_data_max = np.max(test_float_data)
test_float_data_min = np.min(test_float_data)
mu = test_float_data_min
sigma = test_float_data_max - test_float_data_min


# print(float_data.shape)
test_float_data = test_float_data.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1)) #feature_range表示设置归一化范围为（0,1）
test_float_data = scaler.fit_transform(test_float_data)  # 将float_data按列转换为（0,1）之间

test_float_data = np.reshape(test_float_data, (8832, 108))

test_float_data = torch.FloatTensor(test_float_data)


# the input of ST
X_test_input_ST = []


# closeness period
X_test_enc_input = []
# X_test_dec_input = []
#
# X_test_dec_output = []
X_test_output = []

# # day period
X_test_input_day = []
#
# # week period
X_test_input_week = []


# SE

SE_initial = np.load('./Ningxia-YC/SE_all_Ningxia-YC.npy')
# TE = np.load('TE.npy')
# TE = torch.FloatTensor(TE)
# print(SE.shape) # shape (29, 64)
# print(TE.shape) # shape (17000, 295)

SE_test_input = torch.FloatTensor(SE_initial)
SE_test_input = torch.unsqueeze(SE_test_input, dim=0)

# TE

TE_test_input = torch.zeros([988, 17*2])
TE_test_input = torch.FloatTensor(TE_test_input)


for i in range(7832, 8820):
    X_test_enc_input.append(test_float_data[i - 12:i])

    X_test_output.append(test_float_data[i:i+12])


    X_test_input_ST.append(test_float_data[i-5:i])
    # X_test_input.append(test_float_data[i-5:i])
    # X_test_output.append(test_float_data[i])

    X_test_input_day_one_sample = []
    for j in range(i-(12 * 24 * 7), i, 288):
        X_test_input_day_one_sample.append(test_float_data[j])
    X_test_input_day_one_sample = torch.stack(X_test_input_day_one_sample)
    X_test_input_day.append(X_test_input_day_one_sample)

    X_test_input_week_one_sample = []
    for z in range(i-(12 * 24 * 7 * 1), i, (12 * 24 * 7)):
        X_test_input_week_one_sample.append(test_float_data[z])
    X_test_input_week_one_sample = torch.stack(X_test_input_week_one_sample)
    X_test_input_week.append(X_test_input_week_one_sample)


X_test_enc_input = torch.stack(X_test_enc_input)
X_test_input_day = torch.stack(X_test_input_day)
X_test_input_week = torch.stack(X_test_input_week)
X_test_output = torch.stack(X_test_output)

# print(X_test_enc_input.shape) # shape torch.Size([1000, 5, 108])
# print(X_test_input_day.shape) # shape torch.Size([1000, 7, 108])
# print(X_test_input_week.shape) # shape torch.Size([1000, 1, 108])
# print(X_test_output.shape) # shape torch.Size([988, 12, 108])




# the input of ST
# ------------------------------------------------

X_test_input_ST = torch.stack(X_test_input_ST)
# print(X_test_input_ST.shape) # shape torch.Size([1000, 5, 108])
# ------------------------------------------------


# the input of casual diffusion
# ------------------------------------------------
X_test_input_CD = X_test_enc_input
# print(X_test_input_CD.shape) # shape torch.Size([1000, 5, 108])
# ------------------------------------------------




class MyDataset(Data.Dataset):
    def __init__(self, X_train_input_ST, X_train_enc_input, X_train_input_day, X_train_input_week, TE_train_input, X_train_output, X_train_input_CD): # 初始化方法
        self.X_train_input_ST = X_train_input_ST
        self.X_train_enc_input = X_train_enc_input # 在类里面需要跨函数调用的话，需要加上self。加上之后，就可以在同一个类的其他函数之中调用该函数的这个参数了
        self.X_train_input_day = X_train_input_day
        self.X_train_input_week = X_train_input_week
        self.TE_train_input = TE_train_input
        self.X_train_output = X_train_output
        self.X_train_input_CD = X_train_input_CD
    def __getitem__(self, idx): # idx为取出的样本索引，索引取值范围，由def __len__(self):确定
        return self.X_train_input_ST[idx], self.X_train_enc_input[idx], self.X_train_input_day[idx], self.X_train_input_week[idx], self.TE_train_input[idx], self.X_train_output[idx], self.X_train_input_CD[idx]  # 仅需返回一个样本即可
    def __len__(self): # 获取所有数据的长度。输入数据和标签长度相同，获取两个其中一个都可以
        return len(self.X_train_input_ST)


# 设定一个dataloader，从train_dataset中提取数据，一次提取一个batch_size的输入数据和相应的标签
train_loader = DataLoader(MyDataset(X_train_input_ST, X_train_enc_input, X_train_input_day, X_train_input_week, TE_train_input, X_train_output, X_train_input_CD), batch_size=128, shuffle=False) # 在MyDataset中，一次只能取一个样本，但是由于有DataLoader,传入到Mydataset的索引不是一个，而是一个batch


# 设定一个dataloader，从validation_dataset中提取数据，一次提取一个batch_size的输入数据和相应的标签
validation_loader = DataLoader(MyDataset(X_validation_input_ST, X_validation_enc_input, X_validation_input_day, X_validation_input_week, TE_validation_input, X_validation_output, X_validation_input_CD), batch_size=10000, shuffle=False) # 在MyDataset中，一次只能取一个样本，但是由于有DataLoader,传入到Mydataset的索引不是一个，而是一个batch


# 设定一个dataloader，从test_dataset中提取数据，一次提取一个batch_size的输入数据和相应的标签
test_loader = DataLoader(MyDataset(X_test_input_ST, X_test_enc_input, X_test_input_day, X_test_input_week, TE_test_input, X_test_output, X_test_input_CD), batch_size=10000, shuffle=False) # 在MyDataset中，一次只能取一个样本，但是由于有DataLoader,传入到Mydataset的索引不是一个，而是一个batch

