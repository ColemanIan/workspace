import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from hyperparameters import hp



# Custom modules
from CBAM import CBAM
from residual_block import ResidualBlock
from downsample import Downsample
from upsample import Upsample


class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_1  = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.silu    = nn.SiLU()



    def forward(self, ldr_image):
        x = self.conv_1(ldr_image)
        x = self.silu(x)
        return x




class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()


        # ENCODER 
        # Consists of conv+silu block then repeating residual and CBAM blocks with skip connections after the CBAM blocks,
        # then downsampling blocks to reduce the spatial dimensions and increase the number of channels.

        self.first_block   = FirstBlock(in_channels, out_channels, kernel_size)

        # Enc 1
        self.enc1 = nn.Sequential(
            ResidualBlock(out_channels, out_channels, kernel_size),
            CBAM(out_channels, out_channels, kernel_size)
        )
        # Skip connection from first block to after CBAM in enc1
        self.down1 = Downsample(out_channels, out_channels*2, kernel_size)

        # Enc 2
        self.enc2 = nn.Sequential(
            ResidualBlock(out_channels*2, out_channels*2, kernel_size),
            CBAM(out_channels*2, out_channels*2, kernel_size)
        )
        # Skip connection
        self.down2 = Downsample(out_channels*2, out_channels*4, kernel_size)

        # Enc 3
        self.enc3 = nn.Sequential(
            ResidualBlock(out_channels*4, out_channels*4, kernel_size),
            CBAM(out_channels*4, out_channels*4, kernel_size)
        )
        # Skip connection
        self.down3 = Downsample(out_channels*4, out_channels*8, kernel_size)
        
        # Enc 4
        self.enc4 = nn.Sequential(
            ResidualBlock(out_channels*8, out_channels*8, kernel_size),
            CBAM(out_channels*8, out_channels*8, kernel_size)
        )
        # Skip connection
        self.down4 = Downsample(out_channels*8, out_channels*16, kernel_size)

        # Enc 5
        self.enc5 = nn.Sequential(
            ResidualBlock(out_channels*16, out_channels*16, kernel_size),
            CBAM(out_channels*16, out_channels*16, kernel_size)
        )
        # Skip connection
        self.down5 = Downsample(out_channels*16, out_channels*32, kernel_size)

        # DECODER
        # Consists of repeating upsampling blocks with skip connections from the encoder blocks, then a
        # final conv block to reduce the number of channels back to 3 for the output HDR image.
        # Also a contextual attention block to capture long-range dependencies in the feature maps.


        
    def forward(self, ldr_image):
        # Input: LDR image of shape (batch_size, 3, H, W)
        x = self.first_block(ldr_image)

        #skip connection
        skip = x



        

        
        
        
        
        
        return x


# model = AttentionHDR().to(device)

