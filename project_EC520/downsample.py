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

    @staticmethod
    def test():
        print("Testing Downsample...")
        batch_size, in_c, h, w = 2, 32, 64, 64
        out_c = 64
        x = torch.randn(batch_size, in_c, h, w)
        model = Downsample(in_channels=in_c, out_channels=out_c)
        y = model(x)
        assert y.shape == (batch_size, out_c, h // 2, w // 2), f"Downsample shape mismatch: {y.shape}"
        print("Downsample math/shapes test passed!")

if __name__ == '__main__':
    Downsample.test()