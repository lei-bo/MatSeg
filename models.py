import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as T
from torchvision import models
from torchvision.models import vgg16, vgg16_bn, resnet50, resnet152, densenet121, densenet201, inception_v3

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from scipy.misc import imresize
from tqdm import tqdm
from collections import namedtuple

class Resnet50(torch.nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        features = list(resnet50(pretrained=True).children())[:-1]
        self.features = nn.ModuleList(features)
        self.linear = nn.Linear(3904, 4, bias=True)

    def forward(self, x):
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {2, 4, 5, 6, 7}:
                upsample = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=True)
                features.append(upsample)
        outputs = torch.cat(features, 1)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.linear(outputs)
        outputs = outputs.permute(0, 3, 1, 2)
        return outputs

class Resnet152(torch.nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        features = list(resnet152(pretrained=True).children())[:-1]
        self.features = nn.ModuleList(features)
        self.linear = nn.Linear(3904, 4, bias=True)

    def forward(self, x):
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {2, 4, 5, 6, 7}:
                upsample = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=True)
                features.append(upsample)
        outputs = torch.cat(features, 1)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.linear(outputs)
        outputs = outputs.permute(0, 3, 1, 2)
        return outputs


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 1024, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 4, bias=True),
        )
        self.linear = nn.Linear(1472, 4, bias=True)

    def forward(self, x):
        features = []
        size = (x.size(2), x.size(3))
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                #                 print(x.size())
                upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)

                for multiple in range(512 // x.size(1)):
                    features.append(upsample)
        outputs = torch.cat(features, 1)
        #         raise Exception('hahaha')
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.classifier(outputs)
        outputs = outputs.permute(0, 3, 1, 2)
        return outputs

#         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
#         return vgg_outputs(*results)

class Vgg16_6x(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features)
        self.conv7 = vgg16(pretrained=True).classifier[0:5]
        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 4, bias=True),
        )
        #         self.linear = nn.Linear(1472, 4, bias=True)
        self.linear = nn.ModuleList([nn.Linear(64, 512, bias=True),
                                     nn.Linear(128, 512, bias=True),
                                     nn.Linear(256, 512, bias=True),
                                     nn.Linear(512, 512, bias=True),
                                     nn.Linear(512, 512, bias=True),
                                     nn.Linear(4096, 512, bias=True)])

    def forward(self, x):
        features, ind = [], 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                upsample = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=True)
                upsample = upsample.permute(0, 2, 3, 1)
                upsample = self.linear[ind](upsample)
                features.append(upsample)
                ind += 1
        x = x.view(x.size(0), -1)
        x = self.conv7(x)
        x = x.view((*x.size(), 1, 1))
        upsample = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=True)
        upsample = upsample.permute(0, 2, 3, 1)
        upsample = self.linear[-1](upsample)
        features.append(upsample)
        outputs = torch.cat(features, 3)
        outputs = self.classifier(outputs)
        outputs = outputs.permute(0, 3, 1, 2)
        return outputs

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
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
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

class pixelnet_densenet(torch.nn.Module):
    def __init__(self, K=4):
        super(pixelnet_densenet, self).__init__()
        features = list(densenet121(pretrained=True).features)
        self.features = nn.ModuleList(features)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 1024, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, K, bias=True),
        )

    def set_train_flag(self, flag):
        self.train_flag = flag

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def forward(self, x):
        feature_maps_index = {2, 4, 6}#, 8, 10}
        if self.train_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    # print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    for multiple in range(512 // x.size(1)):
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
                    for multiple in range(512 // map.size(1)):
                        features.append(upsample)
                output = torch.cat(features, 1)
                output = output.permute(0, 2, 1)
                output = self.classifier(output)
                output = output.permute(0, 2, 1)
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs

class hypercolumn_vgg(torch.nn.Module):
    def __init__(self):
        super(hypercolumn_vgg, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features)
        # self.linear = nn.Linear(1472, 4, bias=True)

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def set_dense_flag(self, flag):
        self.dense_flag = flag

    def forward(self, x):
        feature_maps_index = {3, 8, 15, 22, 29}
        # feature_maps_index = {3}
        if not self.dense_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    # print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(512 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
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
                output = torch.cat(features, 1).cpu()
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs

class hypercolumn_resnet50(torch.nn.Module):
    def __init__(self):
        super(hypercolumn_resnet50, self).__init__()
        features = list(resnet50(pretrained=True).children())[:-1]
        self.features = nn.ModuleList(features)

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def set_dense_flag(self, flag):
        self.dense_flag = flag

    def forward(self, x):
        if not self.dense_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {2, 4, 5, 6, 7}:
                    # print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(2048 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
        else:
            size, size_prod = (x.size(2), x.size(3)), x.size(2)*x.size(3)
            feature_maps = []
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {2, 4, 5, 6, 7}:
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
                    # for multiple in range(2048 // map.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
                output = torch.cat(features, 1).cpu()
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs

class hypercolumn_resnet152(torch.nn.Module):
    def __init__(self):
        super(hypercolumn_resnet152, self).__init__()
        features = list(resnet152(pretrained=True).children())[:-1]
        self.features = nn.ModuleList(features)

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def set_dense_flag(self, flag):
        self.dense_flag = flag

    def forward(self, x):
        # feature_maps_index = {2, 4, 5, 6, 7}
        feature_maps_index = {3}
        if not self.dense_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                # print(x.size())
                if ii in feature_maps_index:
                    # print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(2048 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
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
                    # for multiple in range(2048 // map.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
                output = torch.cat(features, 1).cpu()
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs

class hypercolumn_densenet121(torch.nn.Module):
    def __init__(self):
        super(hypercolumn_densenet121, self).__init__()
        features = list(densenet121(pretrained=True).features)
        self.features = nn.ModuleList(features)
        # self.linear = nn.Linear(1472, 4, bias=True)

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def set_dense_flag(self, flag):
        self.dense_flag = flag

    def forward(self, x):
        if not self.dense_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {2, 4, 6, 8, 10}:
                    # print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(1024 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
        else:
            size, size_prod = (x.size(2), x.size(3)), x.size(2)*x.size(3)
            feature_maps = []
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {2, 4, 6, 8, 10}:
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
                    # for multiple in range(1024 // map.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
                output = torch.cat(features, 1).cpu()
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs

class hypercolumn_densenet201(torch.nn.Module):
    def __init__(self):
        super(hypercolumn_densenet201, self).__init__()
        features = list(densenet201(pretrained=True).features)
        self.features = nn.ModuleList(features)

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def set_dense_flag(self, flag):
        self.dense_flag = flag

    def forward(self, x):
        feature_maps_index = {2, 4, 6, 8, 10}
        # feature_maps_index = {0,2,4}
        if not self.dense_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in feature_maps_index:
                    # print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(1024 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
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
                    # for multiple in range(1024 // map.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
                output = torch.cat(features, 1).cpu()
                outputs.append(output)
            outputs = torch.cat(outputs, 2)
        return outputs

class hypercolumn_inception(torch.nn.Module):
    def __init__(self):
        super(hypercolumn_inception, self).__init__()
        features = list(inception_v3(pretrained=True).children())[0:13]
        self.features = nn.ModuleList(features)
        # self.linear = nn.Linear(1472, 4, bias=True)

    def set_rand_ind(self, rand_ind):
        self.rand_ind = rand_ind

    def set_dense_flag(self, flag):
        self.dense_flag = flag

    def forward(self, x):
        if not self.dense_flag:
            features = []
            size = (x.size(2), x.size(3))
            for ii, model in enumerate(self.features):
                x = model(x)
                # if ii in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}:
                if ii in {3,7,12}:
                    print(x.size())
                    upsample = F.upsample(x, size=size, mode='bilinear', align_corners=True)
                    upsample = upsample.view(upsample.size(0), upsample.size(1), -1)[:,:,self.rand_ind]
                    # for multiple in range(1024 // x.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
            outputs = torch.cat(features, 1)
        else:
            size, size_prod = (x.size(2), x.size(3)), x.size(2)*x.size(3)
            feature_maps = []
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {3,7,12}:
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
                    # for multiple in range(1024 // map.size(1)):
                    #     features.append(upsample)
                    features.append(upsample)
                output = torch.cat(features, 1).cpu()
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

class unet_dense121(torch.nn.Module):
    def __init__(self, K=4):
        super(unet_dense121, self).__init__()
        # self.down = densenet201(pretrained=True).features
        self.features = nn.ModuleList(list(densenet121(pretrained=False).features))
        self.conv1024 = self.double_conv(512, 512) # 1024,1024 for {2,4,6,8,10}
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
            # self.double_conv(2048, 1024),
            # self.double_conv(2048, 512),
            self.double_conv(1024, 256),
            self.double_conv(512, 64),
            self.double_conv(128, 64),
            self.double_conv(67, 64),
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
        features = [x]
        for ii, model in enumerate(self.features):
            x = model(x)
            # print(x.shape)
            if ii in {2, 4, 6}:
                features.append(x)
                # print(x.shape)
        x = self.conv1024(features[-1])
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


class wnet(torch.nn.Module):
    def __init__(self, K=10):
        super(wnet, self).__init__()
        self.down1 = vgg16(pretrained=True).features
        self.down2 = vgg16(pretrained=True).features
        self.features1 = nn.ModuleList(list(self.down1))
        self.features2 = nn.ModuleList(list(self.down2))
        self.conv1024_1 = self.double_conv(512, 1024)
        self.conv1024_2 = self.double_conv(512, 1024)
        self.conv1 = nn.ModuleList([
            self.double_conv(1536, 512),
            self.double_conv(1024, 512),
            self.double_conv(768, 256),
            self.double_conv(384, 128),
            self.double_conv(192, 64),
        ])
        self.conv2 = nn.ModuleList([
            self.double_conv(1536, 512),
            self.double_conv(1024, 512),
            self.double_conv(768, 256),
            self.double_conv(384, 128),
            self.double_conv(192, 64),
        ])
        self.linear1 = nn.Sequential(
            nn.Linear(64, 128, bias=True),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128, K, bias=True),
        )
        self.conv_mid = nn.Conv2d(K, 3, kernel_size=3, padding=1)
        self.linear2 = nn.Sequential(
            nn.Linear(64, 128, bias=True),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(128, 3, bias=True),
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
        features1, features2 = [], []
        for ii, model in enumerate(self.features1):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                features1.append(x)
        x = self.conv1024_1(x)
        for i in range(len(self.conv1)):
            x = F.upsample(x, size=(features1[-1-i].size(2), features1[-1-i].size(3)), mode='bilinear', align_corners=True)
            x = torch.cat([x, features1[-1-i]], 1)
            x = self.conv1[i](x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear1(x)
        x = x.permute(0, 3, 1, 2)
        seg = F.softmax(x, dim=1)
        x = self.conv_mid(x)
        for ii, model in enumerate(self.features2):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                features2.append(x)
        x = self.conv1024_2(x)
        for i in range(len(self.conv2)):
            x = F.upsample(x, size=(features1[-1-i].size(2), features1[-1-i].size(3)), mode='bilinear', align_corners=True)
            x = torch.cat([x, features2[-1-i]], 1)
            x = self.conv2[i](x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear2(x)
        x = x.permute(0, 3, 1, 2)
        return x, seg

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt
    return torch.from_numpy(weight)

class NcutLoss_old(nn.Module):
    def __init__(self, sigmaI=10, sigmaX=4, r=5, sample_num=1000):
        super(NcutLoss, self).__init__()
        self.sigmaI = sigmaI
        self.sigmaX = sigmaX
        self.r = r
        self.indices, self.expos = self.get_pixel_info()
        self.sample_num = sample_num
    def get_pixel_info(self):
        indices, expos = [], []
        for i in range(-self.r, self.r+1):
            for j in range(-self.r, self.r+1):
                if i**2 + j**2 <= self.r**2:
                    indices.append([i,j])
                    expos.append(np.exp(-(i**2+j**2)/(self.sigmaX**2)))
        indices, expos = np.array(indices), np.array(expos)
        return indices, expos
    def forward(self, predictions, imgs):
        # predictions, imgs = predictions.cpu(), imgs.cpu()
        K, height, width = predictions.size(1), predictions.size(2), predictions.size(3)
        h_sample = np.random.randint(self.r, height-self.r, self.sample_num)
        w_sample = np.random.randint(self.r, width-self.r, self.sample_num)

        numerator, denominator = 0, 0

        for i in range(h_sample.shape[0]):
            p = predictions[:, :, h_sample[i], w_sample[i]]
            indices = np.array([h_sample[i], w_sample[i]]) + self.indices
            weight = np.exp(-np.linalg.norm(imgs[:, :, h_sample[i], w_sample[i]].unsqueeze(2) - imgs[:, :, indices[:, 0], indices[:, 1]], axis=1,
                            keepdims=True) ** 2 / (self.sigmaI ** 2)) * self.expos
            weight = torch.Tensor(weight).cuda()
            q = predictions[:, :, indices[:, 0], indices[:, 1]]
            numerator += (weight * q).sum(2) * p
            denominator += weight.sum(2) * p

        # weights = torch.zeros(height, width, self.indices.shape[0])
        # qs = torch.zeros(predictions.size(0), K, height, width, self.indices.shape[0])
        # for h in range(self.r, height-self.r):
        #     for w in range(self.r, width-self.r):
        #         p = predictions[:,:,h,w]
        #         count += 1
        #         if count % 100000 == 1:
        #             print(count)
        #         indices = np.array([h,w]) + self.indices
                # valid_seq = list(set(np.where(indices[:, 0]>=0)[0])&set(np.where(indices[:,0] <height)[0])&set(np.where(indices[:,1]>=0)[0])&set(np.where(indices[:,1]<width)[0]))
                # indices, expos = indices[valid_seq], self.expos[valid_seq]
                # expos = self.expos
                # weight = np.exp(-np.linalg.norm(imgs[:,:,h,w].unsqueeze(2) - imgs[:,:,indices[:,0],indices[:,1]], axis=1, keepdims=True)**2/(self.sigmaI ** 2))*expos
                # weight = torch.Tensor(weight)
                # weights[h,w,:] = weight
                # q = predictions[:, :, indices[:, 0], indices[:, 1]]
                # qs[:,:,h,w,:] = q

                # q = predictions[:,:,indices[:,0],indices[:,1]]
                # numerator += (weight * q).sum(2) * p
                # denominator += weight.sum(2) * p

        # weights = weights.unsqueeze(0).unsqueeze(0)
        # numerator = ((weights*qs).sum(4)*predictions).sum(3).sum(2)
        # denominator = (weights.sum(4) * predictions).sum(3).sum(2)

        loss = (K - torch.sum(numerator/denominator, dim=1)).sum().cuda()
        return loss

class NcutLoss(nn.Module):
    def __init__(self, sigmaI=10, sigmaX=4, r=5, sample_num=1000):
        super(NcutLoss, self).__init__()
        self.sigmaI = sigmaI
        self.sigmaX = sigmaX
        self.r = r
        self.indices, self.expos, self.filters = self.get_filters()
        self.sample_num = sample_num
    def get_filters(self):
        indices, expos, filters = [], [], []
        for i in range(0, 2*self.r+1):
            for j in range(0, 2*self.r+1):
                filter = np.zeros((self.r*2+1, self.r*2+1))
                if (i-self.r)**2 + (j-self.r)**2 <= self.r**2:
                    indices.append([i,j])
                    expos.append(np.exp(-((i-self.r)**2 + (j-self.r)**2)/(self.sigmaX**2)))
                    filter[i][j] = 1
                    filters.append(filter)
        indices, expos, filters = np.array(indices), np.array(expos), np.array(filters)
        filters = torch.Tensor(filters).unsqueeze(1).cuda()
        expos = torch.Tensor(expos).unsqueeze(1).unsqueeze(1).cuda()
        return indices, expos, filters
    def get_weights(self, features):
        # filters = torch.Tensor(self.filters).unsqueeze(1)
        features = features.unsqueeze(1).cuda()
        filters = self.filters.unsqueeze(1)
        peri_pixels = F.conv3d(features, filters, padding=[0,self.r,self.r])
        weights = torch.exp(-(torch.norm(peri_pixels - features, dim = 2)**2/(self.sigmaI ** 2))*self.expos)
        weights = weights.cuda()
        return weights
    def forward(self, predictions, imgs):
        N, K, height, width = predictions.size(0), predictions.size(1), predictions.size(2), predictions.size(3)
        weights = self.get_weights(imgs)
        # weights = weights.cuda()
        # filters = torch.Tensor(self.filters).unsqueeze(1).cuda()
        loss = K
        for i in range(K):
            pred = predictions[:,i,:,:].unsqueeze(1)
            peri_pred = F.conv2d(pred, self.filters, padding=self.r)
            loss -= (pred*weights*peri_pred).sum()/(pred*weights).sum()
        return loss

if __name__ == '__main__':
    x = torch.Tensor(1,3,768,1024).cuda()
    # model = unet_dense121(K=2).cuda()
    model = unet_full(K=2).cuda()
    # model = unet(K=2).cuda()
    output = model(x)
    print(output.shape)
