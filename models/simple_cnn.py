"""
models/simple_cnn.py
---------------------
Defines the CNN architecture used in Part 1 (training from scratch).

Structure:
- Three convolution blocks (Conv → BatchNorm → ReLU → MaxPool).
- Dropout for regularization.
- Two fully-connected layers for classification into 10 classes.

This file only defines the network structure.
It does NOT handle data loading or training.
"""


import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution blocks
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Convolution + activation + pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        # Dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
