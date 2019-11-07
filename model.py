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

'''-------------Net1(IDSIA3):3Conv(150,200,300) | 2FC-(350,43) | BatchNorm | FCDropOut -------------'''

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 150, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(150)
        self.conv2 = nn.Conv2d(150, 200, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 300, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(300)
        self.fc1 = nn.Linear(300 * 3 * 3, 350)
        self.bn4 = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.conv_drop = nn.Dropout2d(p=0.2)
        self.fc_drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv_drop(self.pool(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_drop(self.pool(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_drop(self.pool(x))
        x = x.view(-1, 300 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.fc_drop(x))
        return F.log_softmax(x, dim=1)


'''-------------Net2(IDSIA4):3Conv(150,200,300) | 2FC-(350,43) | BatchNorm | 1STN-------------'''

class Stn(nn.Module):
    def __init__(self, n1, n2, n3, in_=3):
        # 50, 100, 100
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_, n1, 7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(n1, n2, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.flattenN = n2 * 8 * 8
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.flattenN, n3),
            nn.ReLU(),
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

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.stn = Stn(200, 300, 200)
        self.conv1 = nn.Conv2d(3, 150, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(150)
        self.conv2 = nn.Conv2d(150, 200, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 300, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(300)
        self.fc1 = nn.Linear(300 * 3 * 3, 350)
        self.bn4 =nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.fc_drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 300 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

'''-------------Net3(IDSIA5):3Conv(150,200,300) | 2FC-(350,43) | BatchNorm | FCDropOut | 1STN-------------'''

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.stn = Stn(200, 300, 200)
        self.conv1 = nn.Conv2d(3, 150, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(150)
        self.conv2 = nn.Conv2d(150, 200, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 300, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(300)
        self.fc1 = nn.Linear(300 * 3 * 3, 350)
        self.bn4 =nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.conv_drop = nn.Dropout2d(p=0.2)
        self.fc_drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv_drop(self.pool(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_drop(self.pool(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_drop(self.pool(x))
        x = x.view(-1, 300 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.fc_drop(x))
        return F.log_softmax(x, dim=1)

'''-------------Net4 (GNet2):4Conv(150,200,300,350) | 2FC-(350,43) | BatchNorm | FCDropOut | 2STNs-------------'''

class Stn1(nn.Module):
    def __init__(self, n1, n2, n3, in_channels=3, in_size=48,
                 kernel_size=5, padding=2):
        super(Stn1, self).__init__()
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


class Net4(nn.Module):
    def __init__(self, n1=150, n2=200, n3=300, n4=350, in_channels=3, in_size=48, padding=2):
        super(Net4, self).__init__()
        self.stn_1 = Stn1(200, 300, 200)
        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=7, padding=padding)
        self.bn1 = nn.BatchNorm2d(n1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=4, padding=padding)
        self.bn2 = nn.BatchNorm2d(n2)
        self.stn_2 = Stn1(150, 150, 150, in_channels=n2, in_size=12)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=4, padding=padding)
        self.bn3 = nn.BatchNorm2d(n3)

        self.flattenN = n3 * 6 * 6
        self.fc1 = nn.Linear(self.flattenN, n4)
        self.bn4 = nn.BatchNorm1d(n4)
        self.fc2 = nn.Linear(n4, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn_1(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.stn_2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattenN)
        x = F.relu(self.bn4(self.fc1(x)))
        # x = self.fc2(x)
        x = self.fc2(self.drop(x))
        return F.log_softmax(x, dim=1)

'''-----------Ner5(GNet3):4Conv(150,200,300, 350) | 2FC-(350,43) | BatchNorm | FCDropOut | 2STNs+BN-----------'''

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

class Net5(nn.Module):
    def __init__(self, n1=150, n2=200, n3=300, n4=350, in_channels=3, in_size=48, padding=2):
        super(Net5, self).__init__()
        self.stn_1 = Stn2(200, 300, 200)
        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=7, padding=padding)
        self.bn1 = nn.BatchNorm2d(n1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=4, padding=padding)
        self.bn2 = nn.BatchNorm2d(n2)
        self.stn_2 = Stn2(150, 150, 150, in_channels=n2, in_size=12)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=4, padding=padding)
        self.bn3 = nn.BatchNorm2d(n3)

        self.flattenN = n3 * 6 * 6
        self.fc1 = nn.Linear(self.flattenN, n4)
        self.bn4 = nn.BatchNorm1d(n4)
        self.fc2 = nn.Linear(n4, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn_1(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.stn_2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattenN)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.drop(x))
        return F.log_softmax(x, dim=1)
