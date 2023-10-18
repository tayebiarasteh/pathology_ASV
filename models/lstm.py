"""
Created on Thu Sep 20 16:56:19 2018
@author: Harry Volek

@modified by: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/starasteh/
on March 31, 2020
"""

import torch
import torch.nn as nn
import pdb

from utils.utils import *


class SpeechEmbedder(nn.Module):
    def __init__(self, nmels, hidden_dim=500, output_dim=256, num_layers=1):
        super().__init__()
        self.LSTM_stack = nn.LSTM(nmels, hidden_dim, num_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hidden_dim, output_dim)


    def forward(self, input_tensor):
        input_tensor, _ = self.LSTM_stack(input_tensor.float())  # (batch, frames, n_mels)
        # only use last frame
        input_tensor = input_tensor[:, input_tensor.size(1) - 1]
        input_tensor = self.projection(input_tensor.float())
        # L2 normalization of the network outputs
        input_tensor = input_tensor / torch.norm(input_tensor, dim=1).unsqueeze(1)
        return input_tensor
