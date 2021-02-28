import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from quantize import *
from fft_conv import FFTConv2d

device = "cuda" if torch.cuda.is_available() else "cpu"

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        m = self.in_features
        n = self.out_features
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        out = quant8(out)
        return out
        pass

class PhotonicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PhotonicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=1)
        self.conv = FFTConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
        # Define shot noise
        Nb=8 # 8-bit precision
        self.np=(2/3)*2**(2*Nb) # number of photons (per pixel) required to achieve N_b precision given SNR from shot noise is sqrt(np)
        
        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))

    def forward(self, x):
        out = self.conv(x)

        shot_noise = (1 / np.sqrt(self.np)) * torch.randn(out.size(), device=device)
        out = out + shot_noise

        out = quant8(out)
        return out
