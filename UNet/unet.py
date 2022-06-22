import torch
import torch.nn as nn
from torchvision.models import vgg16

from .unet_parts import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 32)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outconv(x)
        return out


class UNetVgg16(nn.Module):
    def __init__(self, n_classes):
        super(UNetVgg16, self).__init__()
        encoder = vgg16(pretrained=True).features
        self.inc = encoder[:4]
        self.down1 = encoder[4:9]
        self.down2 = encoder[9:16]
        self.down3 = encoder[16:23]
        self.down4 = encoder[23:30]
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.encoder = nn.ModuleList([self.inc, self.down1, self.down2,
                                      self.down3, self.down4])
        self.decoder = nn.ModuleList([self.up1, self.up2, self.up3,
                                      self.up4, self.outc])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out


def test_model(net):
    x = torch.Tensor(1,3,484,645).cuda()
    model = net(n_classes=4).cuda()
    # [print(a, b) for a, b in model.named_modules()]
    output = model(x)
    print(output.shape)


if __name__ == '__main__':
    test_model(UNetVgg16)