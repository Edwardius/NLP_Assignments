#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

### START CODE HERE for part 1d
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, embedding_size):
        super(Highway, self).__init__()

        self.proj_linear = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.gate_linear = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, conv_out):
        proj_linear_out = self.proj_linear(conv_out)
        proj_layer = self.ReLU(proj_linear_out)
        gate_linear_out = self.gate_linear(conv_out)
        gate_layer = self.Sigmoid(gate_linear_out)
        highway_out = proj_layer*gate_layer + (1 - gate_layer)*conv_out

        return highway_out
### END CODE HERE 

