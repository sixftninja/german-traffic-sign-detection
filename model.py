from __future__ import print_function
from collections import namedtuple
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

num_classes = 43

class Stn(nn.Module):
    def __init__(self, n1, n2, n3, in_channels=3, in_size=48,
                 kernel_size=5, padding=2):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size, padding=padding),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n1, n2, kernel_size, padding=padding),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        out_size = (in_size - kernel_size + 1) // 2 + padding
        out_size = (out_size - kernel_size + 1) // 2 + padding
        self.flattenN = n2 * out_size * out_size

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.flattenN, n3),
            nn.BatchNorm1d(n3),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(n3, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, self.flattenN)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

class GNet3(nn.Module):
    def __init__(self, n1=150, n2=200, n3=300, n4=350, in_channels=3, in_size=48, padding=2):
        super(GNet3, self).__init__()
        self.stn1 = Stn(200, 300, 200)
        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=7, padding=padding)
        self.bn1 = nn.BatchNorm2d(n1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=4, padding=padding)
        self.bn2 = nn.BatchNorm2d(n2)
        self.stn3 = Stn(150, 150, 150, in_channels=n2, in_size=12)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=4, padding=padding)
        self.bn3 = nn.BatchNorm2d(n3)

        self.flattenN = n3 * 6 * 6
        self.fc1 = nn.Linear(self.flattenN, n4)
        self.bn4 = nn.BatchNorm1d(n4)
        self.fc2 = nn.Linear(n4, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn1(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.stn3(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattenN)
        x = F.relu(self.bn4(self.fc1(x)))
        # x = self.fc2(x)
        x = self.fc2(self.drop(x))
        return F.log_softmax(x, dim=1)

class Stn2(nn.Module):
    def __init__(self, n1, n2, n3, in_channels=3, in_size=48,
                 kernel_size=5, padding=2):
        super(Stn2, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size, padding=padding),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n1, n2, kernel_size, padding=padding),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        out_size = (in_size - kernel_size + 1) // 2 + padding
        out_size = (out_size - kernel_size + 1) // 2 + padding
        self.flattenN = n2 * out_size * out_size

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.flattenN, n3),
            # nn.BatchNorm1d(n3),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(n3, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, self.flattenN)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x


class GNet2(nn.Module):
    def __init__(self, n1=150, n2=200, n3=300, n4=350, in_channels=3, in_size=48, padding=2):
        super(GNet2, self).__init__()
        self.stn1 = Stn2(200, 300, 200)
        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=7, padding=padding)
        self.bn1 = nn.BatchNorm2d(n1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=4, padding=padding)
        self.bn2 = nn.BatchNorm2d(n2)
        self.stn3 = Stn2(150, 150, 150, in_channels=n2, in_size=12)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=4, padding=padding)
        self.bn3 = nn.BatchNorm2d(n3)

        self.flattenN = n3 * 6 * 6
        self.fc1 = nn.Linear(self.flattenN, n4)
        self.bn4 = nn.BatchNorm1d(n4)
        self.fc2 = nn.Linear(n4, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn1(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.stn3(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattenN)
        x = F.relu(self.bn4(self.fc1(x)))
        # x = self.fc2(x)
        x = self.fc2(self.drop(x))
        return F.log_softmax(x, dim=1)
