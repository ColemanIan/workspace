import torch
import torch.nn as nn

class Upsample(nn.Module):
    """
    Upsampling block using re-size convolution with bi-linear interpolation.

    nn.upsample()
    From doc: The input data is assumed to be of the form minibatch x channels x [optional depth] x [optional height] x width. Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    One can either give a scale_factor or the target output size to calculate the output size. (You cannot give both, as it is ambiguous)


    """

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        )

    def forward(self, x):
        return self.upsample(x)

    @staticmethod
    def test():
        print("Testing Upsample...")
        batch_size, in_c, h, w = 2, 64, 32, 32
        out_c = 32
        x = torch.randn(batch_size, in_c, h, w)
        model = Upsample(in_channels=in_c, out_channels=out_c)
        y = model(x)
        assert y.shape == (batch_size, out_c, h * 2, w * 2), f"Upsample shape mismatch: {y.shape}"
        print("Upsample math/shapes test passed!")

if __name__ == '__main__':
    Upsample.test()