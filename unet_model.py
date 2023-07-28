""" Full assembly of the parts to form the complete network """

from unet_part import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
from torch.cuda import amp
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, T: int, n_channels: int, use_cupy=False, bilinear=True):
        super(UNet, self).__init__()
        
        self.T = T

        if use_cupy:
            functional.set_backend(self, backend='cupy')
        
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 48)
        self.ups = nn.PixelShuffle(4)
        self.smooth = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, image):
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.ups(x)
        x = x.mean(0)                                     # 融合
        x = self.smooth(x) * layer.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(image[0])
        return x
    
    
# 伪训练代码
'''
for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)


            out_fr = net(img)
            loss = F.mse_loss(out_fr, label)
            loss.backward()
            optimizer.step()

            functional.reset_net(net)   # 很重要！！！
'''


# test 

'''
encoder = encoding.PoissonEncoder(step_mode='m')

t = 4

data = torch.zeros(1, 3, 512, 512)

model = UNet(T=t, n_channels=3)

functional.set_step_mode(model, step_mode='m')



data = data.unsqueeze(0).repeat(t, 1, 1, 1, 1)

result = model(encoder(data))

print(result.shape)
'''