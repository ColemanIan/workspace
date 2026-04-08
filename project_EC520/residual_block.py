import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block as defined in LDR->HDR autoencoder paper
    """
    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()
        self.res_conv1   = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.silu        = nn.SiLU()
        self.res_conv2   = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x_              = self.res_conv1(x)        
        x_              = self.silu(x_)
        x_              = self.res_conv2(x_)
        return (x + x_)

    @staticmethod
    def test():
        print("Testing ResidualBlock...")
        batch_size, c, h, w = 2, 32, 64, 64
        x = torch.randn(batch_size, c, h, w)
        model = ResidualBlock(in_channels=c, out_channels=c, kernel_size=3)
        y = model(x)
        assert y.shape == x.shape, f"ResidualBlock shape mismatch: {y.shape} != {x.shape}"
        print("ResidualBlock math/shapes test passed!")

if __name__ == '__main__':
    ResidualBlock.test()