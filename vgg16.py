import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from PhotonicLayers import *

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            PhotonicConv(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PhotonicConv(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PhotonicConv(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifer = nn.Sequential(
            QuantLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            QuantLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            QuantLinear(512, 10)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(3).mean(2)
        out = self.classifer(feat)
        return out

class VGG16_half(nn.Module):
    def __init__(self):
        super(VGG16_half, self).__init__()
        self.features = nn.Sequential(
            PhotonicConv(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            PhotonicConv(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PhotonicConv(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PhotonicConv(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PhotonicConv(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifer = nn.Sequential(
            QuantLinear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            QuantLinear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            QuantLinear(256, 10)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(3).mean(2)
        out = self.classifer(feat)
        return out


class VGG16_5(nn.Module):
    def __init__(self):
        super(VGG16_5, self).__init__()
        self.features = nn.Sequential(
            PhotonicConv(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PhotonicConv(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PhotonicConv(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PhotonicConv(128, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PhotonicConv(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PhotonicConv(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PhotonicConv(256, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PhotonicConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PhotonicConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            )
        self.classifer = nn.Sequential(
            QuantLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            QuantLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            QuantLinear(512, 10)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(3).mean(2)
        out = self.classifer(feat)
        return out
