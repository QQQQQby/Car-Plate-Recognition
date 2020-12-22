# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F


class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        """1 * 20 * 20"""
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        """64 * 10 * 10"""
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        """128 * 5 * 5"""
        self.fc1 = nn.Linear(128 * 5 * 5, 1024, bias=False)
        self.dropout1 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)
        """num_classes"""

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

