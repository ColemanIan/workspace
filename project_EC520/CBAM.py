import torch
from torch import nn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) implementation.

    Given an intermediate feature map, our module sequentially infers attention maps 
    along two separate dimensions, channel and spatial, then the attention maps are 
    multiplied to the input feature map for adaptive feature refinement.

    SiLU activation function is used instead of ReLU per "An Attention-Based Network for Single Image Reconstruction".
    11x11 Convolution is used for spatial attention for the same reasons.
    


    https://arxiv.org/pdf/1807.06521
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # Channel Attention
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.max_pool    = nn.AdaptiveMaxPool2d(1)
        self.mlp         = nn.Sequential(
                        nn.Linear(in_channels, in_channels // 16),
                        nn.SiLU(),
                        nn.Linear(in_channels // 16, in_channels)
                    )
        self.sigmoid     = nn.Sigmoid()

        # Spatial Attention
        self.conv_spat    = nn.Conv2d(2, 1, kernel_size=7, padding=3)
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
        spatial_attn    = self.sigmoid(self.conv_spat(torch.cat([avg_out, max_out], dim=1)))
        x               = x * spatial_attn

        return x

    @staticmethod
    def test():
        print("Testing CBAM...")
        batch_size, channels, h, w = 2, 64, 32, 32
        x = torch.randn(batch_size, channels, h, w)
        model = CBAM(in_channels=channels, out_channels=channels, kernel_size=7)
        y = model(x)
        assert y.shape == x.shape, f"CBAM shape mismatch: {y.shape} != {x.shape}"
        print("CBAM math/shapes test passed!")

if __name__ == '__main__':
    CBAM.test()

