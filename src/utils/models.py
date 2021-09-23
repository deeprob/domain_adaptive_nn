#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TFPerceptron(nn.Module):
    
    def __init__(self, args):
        
        super(TFPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=args.num_features,
                            out_features=1)
        
    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(torch.flatten(x_in, start_dim=1)).squeeze()
        
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        
        return y_out


class TFMLP(nn.Module):
    
    def __init__(self, input_features=32, fc1_nodes=1024, dropout_prob=0.5):        
        super(TFMLP, self).__init__()
        self.fclayers = nn.Sequential(
                         nn.Linear(input_features, fc1_nodes),
                         nn.ReLU(),
                         nn.Dropout(dropout_prob),
                         nn.Linear(fc1_nodes, fc1_nodes//2),
                         nn.Sigmoid(),
                         nn.Linear(fc1_nodes//2, 1)
                    )
        
    def forward(self, x_in, apply_sigmoid=False):
        
        y_out = self.fclayers(torch.flatten(x_in, start_dim=1)).squeeze()
        
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out


class TFMLP_(nn.Module):
    """TFMLP base class for cogan model"""
    def __init__(self, input_features=32, fc1_nodes=1024, dropout_prob=0.5):        
        super(TFMLP_, self).__init__()
        self.fclayers = nn.Sequential(
                         nn.Linear(input_features, fc1_nodes),
                         nn.ReLU(),
                         nn.Dropout(dropout_prob),
                         nn.Linear(fc1_nodes, fc1_nodes//2),
                         nn.Sigmoid(),
                         nn.Linear(fc1_nodes//2, fc1_nodes//4)
                    )
        
    def forward(self, x_in):
        
        y_out = self.fclayers(torch.flatten(x_in, start_dim=1)).squeeze()
        return y_out


class TFSLP(nn.Module):
    """TFSLP base class for cogan model"""
    def __init__(self, input_features=256):        
        super(TFSLP, self).__init__()
        self.fclayers = nn.Sequential(
                         nn.Linear(input_features, 1)
                    )
        
    def forward(self, x_in, apply_sigmoid=False):
        
        y_out = self.fclayers(torch.flatten(x_in, start_dim=1)).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out


class TFCNN(nn.Module):
    
    def __init__(self, channels=4, conv_filters=240, conv_kernelsize=20,
                 maxpool_size=15, maxpool_strides=15):
        super(TFCNN, self).__init__()
        self.featurizer = nn.Sequential(
                            nn.Conv1d(channels, conv_filters, conv_kernelsize, padding="same"),
                            nn.ReLU(),
                            nn.MaxPool1d(maxpool_size, maxpool_strides)
        )
        
    def forward(self, x_in):
        y_out = self.featurizer(x_in)
        
        return y_out


class TFLSTM(nn.Module):
    
    def __init__(self, input_features=240, lstm_nodes=32, fc1_nodes=1024):        
        super(TFLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, lstm_nodes, batch_first=True)
        self.fclayers = TFMLP(input_features=lstm_nodes, fc1_nodes=fc1_nodes, dropout_prob=0.5)
        
    def forward(self, x_in, apply_sigmoid=False):
        # transpose input: (batch_size, filters, seq) -> (batch_size, seq, filters) 
        x_in = torch.transpose(x_in, 1, 2)
        out, (hn,cn) = self.lstm(x_in)
        x_in = out[:, -1, :]
        y_out = self.fclayers(x_in).squeeze()
        
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out


class TFLSTM_(nn.Module):
    """lstm basic class for cogan model"""
    def __init__(self, input_features=240, lstm_nodes=32):        
        super(TFLSTM_, self).__init__()
        self.lstm = nn.LSTM(input_features, lstm_nodes, batch_first=True)
        
    def forward(self, x_in):
        # transpose input: (batch_size, filters, seq) -> (batch_size, seq, filters) 
        x_in = torch.transpose(x_in, 1, 2)
        out, (hn,cn) = self.lstm(x_in)
        x_in = out[:, -1, :]
        return x_in
