import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn, resnet50, resnet152, densenet121, densenet201, inception_v3


class pixelnet(torch.nn.Module):
    def __init__(self, K=4):
        super(pixelnet, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features)
        self.classifier = nn.Sequential(
            nn.Linear(1472, 2048, bias=True),#2560
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, K, bias=True),
        )
        self.linear = nn.Linear(1472, K, bias=True)

    def set_train_flag(self, flag):
        self.train_flag = flag

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def forward(self, x):
        feature_maps_index = {3, 8, 15, 22, 29}
        if self.train_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    # print(x.size())
                    upsample = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(512 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
            #         raise Exception('hahaha')
            outputs = outputs.permute(0, 2, 1)
            outputs = self.classifier(outputs)
            outputs = outputs.permute(0, 2, 1)
        else:
            size, size_prod = (x.size(2), x.size(3)), x.size(2)*x.size(3)
            feature_maps = []
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    feature_maps.append(x)
            outputs = []
            for ind in range(0, size_prod, 10000):
                if ind + 10000 > size_prod:
                    ind_range = range(ind, size_prod)
                else:
                    ind_range = range(ind, ind+10000)
                features = []
                for map in feature_maps:
                    upsample = F.upsample(map, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:, :, ind_range]
                    # for multiple in range(512 // map.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
                output = torch.cat(features, 1)
                output = output.permute(0, 2, 1)
                output = self.classifier(output)
                output = output.permute(0, 2, 1)
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs


class unet(torch.nn.Module):
    def __init__(self, K=4):
        super(unet, self).__init__()
        self.down = vgg16(pretrained=True).features
        self.features = nn.ModuleList(list(self.down))
        self.conv1024 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # self.upsample = nn.ModuleList([
        #     nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # ])
        # for i in range(len(self.upsample)):
        #     size = self.upsample[i].weight.size()
        #     self.upsample[i].weight.data = bilinear_kernel(size[0], size[1], size[2])
        # self.conv = nn.ModuleList([
        #     self.double_conv(1024, 512),
        #     self.double_conv(1024, 512),
        #     self.double_conv(512, 256),
        #     self.double_conv(256, 128),
        #     self.double_conv(128, 64),
        # ])
        self.conv = nn.ModuleList([
            self.double_conv(1536, 512),
            self.double_conv(1024, 512),
            self.double_conv(768, 256),
            self.double_conv(384, 128),
            self.double_conv(192, 64),
        ])
        # self.linear = nn.Linear(64, 4, bias=True)
        self.linear = nn.Sequential(
            nn.Linear(64, 256, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, K, bias=True),
        )
    def double_conv(self, in_channel, out_channel):
        net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return net

    def forward(self, x):
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                features.append(x)
                # print(x.shape)
        x = self.conv1024(x)
        for i in range(len(self.conv)):
            x = F.interpolate(x, size=(features[-1-i].size(2), features[-1-i].size(3)), mode='bilinear', align_corners=True)
            # x = self.upsample[i](x)
            x = torch.cat([x, features[-1-i]], 1)
            x = self.conv[i](x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        return x


class unet_full(torch.nn.Module):
    def __init__(self, K=4):
        super(unet_full, self).__init__()
        self.conv = nn.ModuleList([
            self.double_conv(3, 64),
            self.double_conv(64, 128),
            self.double_conv(128, 256),
            self.double_conv(256, 512),
        ])
        self.linear = nn.Linear(512, K, bias=True)
    def double_conv(self, in_channel, out_channel):
        net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return net
    def forward(self, x):
        for op in self.conv:
            x = op(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':
    x = torch.Tensor(1,3,768,1024).cuda()
    # model = unet_dense121(K=2).cuda()
    model = unet_full(K=2).cuda()
    # model = unet(K=2).cuda()
    output = model(x)
    print(output.shape)
