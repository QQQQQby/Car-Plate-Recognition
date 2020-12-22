# coding: utf-8

import torch
from torch import nn


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        """1 * 20 * 20"""
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """32 * 10 * 10"""
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """64 * 5 * 5"""
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(1024, 65, bias=True)
        )
        """65"""

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        """1 * 20 * 20"""
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """32 * 10 * 10"""
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """64 * 5 * 5"""
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(1024, 100, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(100, 65, bias=True)
        )
        """65"""

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
