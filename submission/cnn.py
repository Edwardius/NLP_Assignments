#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

### START CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, embedding_size, max_word_size, stride=1, k=5):
        super(CNN, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=k, stride=stride, bias=True)
        self.ReLU = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(kernel_size=max_word_size-k+1, stride=stride)
    def forward(self, x_reshaped):
        x_conv = self.conv_layer(x_reshaped)
        conv_out = self.max_pooling(self.ReLU(x_conv))
        return conv_out


### END CODE HERE

