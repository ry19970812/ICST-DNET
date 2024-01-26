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

class the_main_model(nn.Module):
    def __init__(self):
        super(the_main_model, self).__init__()
        self.projection_pred = nn.Linear(d_model, target_sensor)

        self.ST_Traffic = SFPR()

        self.STCL = STCL_module()

        self.ST_module_weight = nn.Parameter(torch.empty(1).normal_(mean=0.5, std=0.5))

    def forward(self, ST_input, TE_input, SE_input, Y_output, CD_input):

        SE_input = torch.squeeze(SE_input, dim=0)

        # ST module
        X_ST_output = self.ST_Traffic(ST_input, SE_input, TE_input)

        # # STCL module
        # # print(CD_input.shape) # shape torch.Size([128, 5, 108])
        X_STCL_output = self.STCL(CD_input, TE_input, SE_input)
        X_STCL_output = X_STCL_output.view(-1, X_STCL_output.shape[1], X_STCL_output.shape[2] * X_STCL_output.shape[3])
        # print(X_STCL_output.shape) # shape torch.Size([128, 12, 108])

        # # prediction layer
        X_output = self.ST_module_weight * X_ST_output + self.IT_module_weight * X_IT_output
        X_output = self.ST_module_weight * X_ST_output + (1-self.ST_module_weight) * X_STCL_output
        return X_output, Y_output

