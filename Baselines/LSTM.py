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

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dimension, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_out)
        self.linear_network = torch.nn.Linear(hidden_size, num_sites * delay)
    def forward(self, LSTM_Input):
        LSTM_Input = LSTM_Input
        LSTM_Output, (h_n, c_n) = self.lstm(LSTM_Input)
        # print(LSTM_Output.shape) # torch.Size([128, 12, 64]) (batch_size, history_timesteps, features(hidden_size=64))
        LSTM_Output = LSTM_Output[:, -1, :]
        LSTM_Output = self.linear_network(LSTM_Output)
        return LSTM_Output




def train_loop(dataloader, model, loss_fn, optimizer):
    num_loss = []
    size = len(dataloader.dataset)
    for batch, (X_Input, y_Output) in enumerate(dataloader):
        X_Input = X_Input.to(device)
        y_Output = y_Output.to(device)
        outputs = model(X_Input)
        label = y_Output
        loss = loss_fn(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        num_loss.append(loss)
    print(f"train_loss:{num_loss[-1]:>4f}") 

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (X_Input, y_Output) in enumerate(
                dataloader):
            X_Input = X_Input.to(device)
            y_Output = y_Output.to(device)
            outputs = model(X_Input)
            label = y_Output
            loss = loss_fn(outputs, label)
            print(f"val_loss:{loss}")
            return loss

model = LSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss().to(device)
epochs = 128
for epoch in range(epochs):
    print(f"Epoch {epoch+1} \n-----------------------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    val_loss = test_loop(validation_loader, model, loss_fn)
    # if val_loss <= 0.00025:
    #     break
print("Training Done!")

# # 保存整个训练好的模型结构
torch.save(model, '......')
print("Save Done!")
#
# # 调用保存好的模型结构用于最终测试
model = torch.load('......')
model.eval()
with torch.no_grad():
    for X_Input, y_obeserved_Output in test_loader:
        X_Input = X_Input.to(device)
        y_obeserved_Output = y_obeserved_Output.to(device)
        predicts_value = model(X_Input)
    # 逆归一化标准化
        observed_value = y_obeserved_Output
        # observed_value = observed_value.view(-1, feature_timesteps * final_outputs)
        observed_value = observed_value*sigma + mu
        # predicts_value = predicts_value.view(-1, feature_timesteps * final_outputs)
        predicts_value = predicts_value*sigma + mu

        observed_value = observed_value.cpu().detach().numpy()
        predicts_value = predicts_value.cpu().detach().numpy()

        # # 计算MAE、RMSE
        RMSE = math.sqrt(mean_squared_error(observed_value, predicts_value))
        print(f"RMSE:{RMSE}")
        MAE = mean_absolute_error(observed_value, predicts_value)
        print(f"MAE:{MAE}")
        observed_value = np.reshape(observed_value, (-1,))
        predicts_value = np.reshape(predicts_value, (-1,))

        np.save('......', predicts_value)
        np.save('......', observed_value)

