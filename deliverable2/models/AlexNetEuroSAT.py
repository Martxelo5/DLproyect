import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetEuroSAT(nn.Module):
    """
    Custom AlexNet-like architecture for 224x224 images,
    derived from AlexNetCIFAR but with a standard ImageNet stem.
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(AlexNetEuroSAT, self).__init__()
        
        # ---------- Feature extractor ----------
        # Block 1 – large kernel & stride to reduce size quickly
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(384)
        
        # Block 4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        
        # Block 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # ---------- Classifier ----------
        # After all convs: 256 x 6 x 6 = 9216
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Weight initialisation (will be customised externally)
        self._initialize_weights()

    def _initialize_weights(self):
        # Placeholder – actual init will be applied by the training script
        pass

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # 55 -> 27
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))   # 27 -> 13
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))               # 13
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))               # 13
        # Block 5
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))   # 13 -> 6
        
        x = x.view(x.size(0), -1)          # 256 * 6 * 6
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x