import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(AlexNetCIFAR, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        # Block 2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(192)
        # Block 3
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(384)
        # Block 4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        # Block 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Weight initialization (will be applied later based on trial)
        self._initialize_weights()

    def _initialize_weights(self):
        # This will be overridden after construction; we'll call a custom init later
        pass

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x