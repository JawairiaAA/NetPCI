# EA model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# model
# residual conv layers
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()

        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels
                     else nn.Identity()
                    )

    def forward(self, x):

        identity = self.skip(x)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = x + identity

        return torch.relu(x)
        
class Bottleneck(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)        
        
# ------------------------------------------------------------------
# U-Net
# ------------------------------------------------------------------
class netpci(nn.Module):
    def __init__(self, in_channels=10, out_channels=1):
        super(netpci, self).__init__()

        # ====================
        # Encoder
        # ====================
        self.down1 = ConvLayer(in_channels, 8)
        self.down1_pool = nn.MaxPool2d(2)
        self.down2 = ConvLayer(8, 16)
        self.down2_pool = nn.MaxPool2d(2)
        self.down3 = ConvLayer(16, 32)
        self.down3_pool = nn.MaxPool2d(2)

        # ====================
        # Bottleneck- normalize this layer only
        # ====================
        self.center = Bottleneck(32)

        # ====================
        # Decoder
        # ====================
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.up3_conv = ConvLayer(64, 32)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1))
        self.up2_conv = ConvLayer(32, 16)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.up1_conv = ConvLayer(16, 8)
        
        # refine prob fields
        self.refine = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),nn.ReLU(inplace=True))
        
        # Output
        self.output = nn.Conv2d(8, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        # ====================
        # Encoder
        # ====================
        down1 = self.down1(inputs)
        down1_pool = self.down1_pool(down1)
        down2 = self.down2(down1_pool)
        down2_pool = self.down2_pool(down2)
        down3 = self.down3(down2_pool)
        down3_pool = self.down3_pool(down3)

        # ====================
        # Bottleneck
        # ====================
        center = self.center(down3_pool)
        #center = self.center2(center)

        # ====================
        # Decoder
        # ====================
        up3 = self.up3(center)
        up3 = torch.cat([down3, up3], dim=1)
        up3 = self.up3_conv(up3)
        up2 = self.up2(up3)
        up2 = torch.cat([down2, up2], dim=1)
        up2 = self.up2_conv(up2)
        up1 = self.up1(up2)
        up1 = torch.cat([down1, up1], dim=1)
        up1 = self.up1_conv(up1)
        up1 = self.refine(up1)
        
        output = self.output(up1)
        output = self.sigmoid(output)

        return output
        