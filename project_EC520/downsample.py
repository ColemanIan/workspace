import torch
import torch.nn as nn


class Downsample(nn.Module):
    """
    Downsampling block using strided convolution.
    
    Each downsampling block reduces the spatial dimensions of the feature maps 
    by a factor of 2, while increasing the number of channels to capture more complex features.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.downsample(x)