import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.res_conv1   = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.silu        = nn.SiLU()
        self.res_conv2   = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x_              = self.res_conv1(x)        
        x_              = self.silu(x_)
        x_              = self.res_conv2(x_)
        return (x + x_)