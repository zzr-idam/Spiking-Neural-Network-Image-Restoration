""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.cuda import amp
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        functional.set_step_mode(self, step_mode='m')
        self.double_conv = nn.Sequential(
            layer.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(mid_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            
            layer.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan())        
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        self.maxpool_conv = nn.Sequential(
            layer.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        # if bilinear, use the normal convolutions to reduce the number of channelss
        if bilinear:
            self.up = layer.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        sto = torch.zeros((x1.shape[0], x1.shape[1], x1.shape[2] + x2.shape[2], x2.shape[3], x1.shape[4] * 2)).cuda()
        for t in range(x1.shape[0]):
            x11 = self.up(x1[t])
            # input is CHW
            diffY = x2[t].size()[2] - x11.size()[2]
            diffX = x2[t].size()[3] - x11.size()[3]

            x11 = F.pad(x11, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            #print(x2[t].shape)
            #print(x11.shape)
            #print(sto.shape)
            sto[t] = torch.cat([x2[t], x11], dim=1)
        return self.conv(sto)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)