import torch.nn as nn
import torch
from torch.nn import functional as F


class CLDNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.layer1 = nn.Conv1d(1, 128, 3, padding=1)
        self.maxpool = nn.MaxPool1d(2)
        self.layer2 = nn.Conv1d(128, 64, 3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1216, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, n_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.maxpool(x)
        x = F.relu(self.layer2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.output(x)
        return x
