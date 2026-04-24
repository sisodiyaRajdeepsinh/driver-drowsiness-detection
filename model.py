import torch
import torch.nn as nn
import torch.nn.functional as F

class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        # Input image: Grayscale 24x24 (standard for these datasets like MRL)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of the flattened features matching the input 24x24
        # 24x24 -> conv1 -> 22x22 -> pool -> 11x11
        # 11x11 -> conv2 -> 9x9 -> pool -> 4x4
        # 4x4 -> conv3 -> 2x2 -> pool -> 1x1 
        # So 128 channels * 1 * 1 = 128
        self.fc1 = nn.Linear(128 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 2) # 2 classes: Open, Closed

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = self.pool(F.relu(self.conv3(x), inplace=True))
        
        x = x.view(-1, 128 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
