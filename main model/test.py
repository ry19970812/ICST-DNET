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

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (X_ST_input, TE_input, X_output, X_CD) in enumerate(dataloader):
            # 计算损失以及预测
            X_ST_input = X_ST_input.to(device)
            TE_input = TE_input.to(device)
            X_output = X_output.to(device)
            X_CD = X_CD.to(device)
            SE_validation_input = SE_validation_input.to(device)
            final_output, Y_output = model(X_ST_input, TE_input, SE_validation_input, X_output, X_CD)
            label = Y_output
            # final_output = final_output[:, history_timestep:, :]
            # label = label[:, history_timestep:, :]
            # print('final_output:', final_output.shape)
            # print('label:', label.shape)
            # final_output = final_output.view(-1, final_output.shape[1] * final_output.shape[2])
            # label = label.view(-1, label.shape[1] * label.shape[2])
            # print('final_output:', final_output.shape)
            # print('label:', label.shape)
            loss = loss_fn(final_output, label)
            print(f"val_loss:{loss}")

            return loss
