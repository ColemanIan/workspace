import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from hyperparameters import hp

class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_1  = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.silu    = nn.SiLU()



    def forward(self, ldr_image):
        x = self.conv_1(ldr_image)
        x = self.silu(x)
        return x


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



class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) implementation.

    Given an intermediate feature map, our module sequentially infers attention maps 
    along two separate dimensions, channel and spatial, then the attention maps are 
    multiplied to the input feature map for adaptive feature refinement.

    https://arxiv.org/pdf/1807.06521
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # Channel Attention
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.max_pool    = nn.AdaptiveMaxPool2d(1)
        self.mlp         = nn.Sequential(
                        nn.Linear(in_channels, in_channels // 16),
                        nn.ReLU(),
                        nn.Linear(in_channels // 16, in_channels)
                    )
        self.sigmoid     = nn.Sigmoid()

        # Spatial Attention
        self.conv_7x7    = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out         = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out         = self.mlp(self.max_pool(x).view(x.size(0), -1))
        channel_attn    = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x               = x * channel_attn

        # Spatial Attention
        avg_out         = torch.mean(x, dim=1, keepdim=True)
        max_out, _      = torch.max(x, dim=1, keepdim=True)
        spatial_attn    = self.sigmoid(self.conv_7x7(torch.cat([avg_out, max_out], dim=1)))
        x               = x * spatial_attn

        return x




# model = AttentionHDR().to(device)

