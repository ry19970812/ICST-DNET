import torch.nn.functional as F
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import os
import torch.optim as optim
import torch.nn as nn
import argparse
import copy
import random

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x



class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D=D, bn_decay=bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[SE_initial_dimension, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0).to(device)
        # print(SE.shape)
        SE = self.FC_se(SE)
        # print(SE.shape)
        # # # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = torch.FloatTensor(TE).to(device)
        # print(TE.shape)
        TE = self.FC_te(TE)
        # print(TE.shape)
        del dayofweek, timeofday
        return SE + TE

class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K=K, d=d, bn_decay=bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # print(X.shape)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_xst = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.BN = nn.BatchNorm2d(history_timestep)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        # num_feature = XS.shape[1]
        # print(XS.shape)
        # print(XT.shape) 
        XS = self.BN(XS)
        XT = self.BN(XT)
        # print(XS.shape)
        # print(XT.shape)
        XST = self.FC_xst(torch.mul(XS, XT))
        z = torch.sigmoid(torch.add(XST, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H



class similarAttentionAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K=K, d=d, bn_decay=bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_pred, STE_his):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X




class SFPR(nn.Module):
    def __init__(self, K=K, d=d, bn_decay=bn_decay):
        super(SFPR, self).__init__()
        self.d = d
        self.K = K
        D = K * d
        self.STEmbedding = STEmbedding()
        # self.ST_Att = STAttBlock()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=False)
        self.gatedFusion = gatedFusion(K * d, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.transformAttention = similarAttention()

        # self.nonlinear = nn.Linear(the_whole_sensor, 1)




    def forward(self, X_input, X_input_SE, X_input_TE):
        # STE Embedding
        X_input_TE = X_input_TE.view(-1, history_timestep + predicted_timestep, 2)
        # print(X_input_SE.shape)
        STE = self.STEmbedding(X_input_SE, X_input_TE)
        # print(STE.shape) # shape torch.Size([64, 11, 207, 8])
        His_STE = STE[:, :history_timestep, :, :]
        # print(His_STE.shape) # shape torch.Size([64, 10, 207, 8])
        Pred_STE = STE[:, history_timestep:, :, :]

        Spatial_Residual = []
        X_input = torch.unsqueeze(X_input, -1)
        X_input = self.FC_1(X_input) # feature统一成D_dimension
        # print(X_input.shape) # [batch_size, num_step, num_vertex, feature = 64 (1->64)] shape torch.Size([64, 10, 207, 8])
        # # print(His_STE.shape)
        Spatial_Single_Residual = self.spatialAttention(X_input, His_STE)
        # print(Spatial_Single_Residual.shape) # shape torch.Size([64, 10, 207, 8])
        Spatial_Residual.append(Spatial_Single_Residual)
        for i in range(num_Spatial_Residual):
            Spatial_Single_Residual = self.spatialAttention(Spatial_Residual[i], His_STE)
            # print(ST_Single_Residual.shape)
            Spatial_Residual.append(Spatial_Single_Residual)
        # print(len(Spatial_Residual))
        spatial_output = Spatial_Residual[num_Spatial_Residual]
        # print(spatial_output.shape) # shape torch.Size([64, 10, 207, 8])

        Temporal_Residual = []
        Temporal_Single_Residual = self.temporalAttention(X_input)
        Temporal_Residual.append(Temporal_Single_Residual)
        for j in range(num_Temporal_Residual):
            Temporal_Single_Residual = self.temporalAttention(Temporal_Residual[j])
            Temporal_Residual.append(Temporal_Single_Residual)
        temporal_output = Temporal_Residual[num_Temporal_Residual]
        # print(temporal_output.shape) # shape torch.Size([64, 10, 207, 8])

        ST_Fusion = self.gatedFusion(spatial_output, temporal_output)
        Pred_STE = self.Pred_STE_embedding(X_input_future_TE, X_input_future_SE)
        TransAtt_output = self.similarAttention(ST_Fusion, Pred_STE, His_STE)
        TransAtt_output = self.FC_2(TransAtt_output)
        TransAtt_output = TransAtt_output.contiguous().view(TransAtt_output.shape[0], TransAtt_output.shape[1], TransAtt_output.shape[2] * TransAtt_output.shape[3])
        Decoder_ST_Residual = []
        Decoder_ST_Single_Residual = self.ST_Att(TransAtt_output, Pred_STE)
        Decoder_ST_Residual.append(Decoder_ST_Single_Residual)
        for j in range(num_ST_Residual_Decoder):
            Decoder_ST_Single_Residual = self.ST_Att(Decoder_ST_Residual[j], Pred_STE)
            Decoder_ST_Residual.append(Decoder_ST_Single_Residual)
        
        decoder_output = Decoder_ST_Residual[num_ST_Residual_Decoder]
        decoder_output = self.FC_2(decoder_output)
        decoder_output = decoder_output.view(decoder_output.shape[0], decoder_output.shape[1] * decoder_output.shape[2] * decoder_output.shape[3])
        decoder_output = self.FC_2(decoder_output)
        final_output = decoder_output.view(-1, decoder_output.shape[1] * decoder_output.shape[2] * decoder_output.shape[3])
        final_output = self.nonlinear(final_output)
        return final_output
