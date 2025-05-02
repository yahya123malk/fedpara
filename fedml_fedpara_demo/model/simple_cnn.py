import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # Output: 16x28x28
        self.pool = nn.MaxPool2d(2, 2)                            # Output: 16x14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: 32x14x14
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 → relu → pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 → relu → pool → 32x7x7
        x = x.view(-1, 32 * 7 * 7)            # Flatten
        x = F.relu(self.fc1(x))               # FC1
        x = self.fc2(x)                       # FC2
        return x
