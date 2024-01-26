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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def train_loop(dataloader, model, loss_fn, optimizer):
    num_loss = []
    size = len(dataloader.dataset)
    # print(f"size:{size}") # size-一个dataset中sample总数量 9592
    for batch, (X_ST_input, X_closeness, X_day, X_week, TE_input, X_output, X_CD) in enumerate(dataloader):
        # 计算损失以及预测
        X_ST_input = X_ST_input.to(device)
        X_closeness = X_closeness.to(device)
        X_day = X_day.to(device)
        X_week = X_week.to(device)
        TE_input = TE_input.to(device)
        X_output = X_output.to(device)
        X_CD = X_CD.to(device)
        # print(X_output.shape) # shape torch.Size([128, 5, 1])
        # print(X_TE.shape) # shape torch.Size([128, 12])
        final_output, Y_output = model(X_ST_input, X_closeness, X_day, X_week, TE_input, SE_train_input, X_output, X_CD)
        # print('final_output:', final_output.shape)
        label = Y_output
        # print('final_output:', final_output.shape)
        # print('label:', label.shape)
        loss = loss_fn(final_output, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"batch:{batch}")
        loss = loss.item()
        num_loss.append(loss)
    print(f"train_loss:{num_loss[-1]:>4f}") # 一个epoch输出一个train_loss
    return loss
