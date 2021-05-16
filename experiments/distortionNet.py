import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn


class DistortionNet(nn.Module):
  def __init__(self):
    super(DistortionNet, self).__init__()
    self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2),
                             nn.Conv2d(32, 16, kernel_size=3, stride=1),
                             nn.MaxPool2d(2, stride=2),
                             nn.Conv2d(16, 1, kernel_size=1, stride=1),
                             nn.MaxPool2d(2, stride=2))
    self.classifier = nn.Linear(729, 3)
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x