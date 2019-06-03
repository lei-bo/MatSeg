import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.datasets as dsets
from torchvision import models
from torchnet import meter
from torchvision.models import vgg16, vgg16_bn, resnet50, resnet152

import os
import scipy.io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize
from tqdm import tqdm
from collections import namedtuple

from utils import *
from models import *
from config import get_config


class Dataset(data.Dataset):
    def __init__(self, root, size, transform_img, transform_label):
        self.img_root = root + "/images/"
        self.label_root = root + "/labels_npy/"
        self.img_names = os.listdir(self.img_root)
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_dir = self.img_root + img_name
        label_dir = "%s%s.npy" % (self.label_root, img_name[:-4])# the name of the label numpy file
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform_img(img)
        label = np.load(label_dir).astype(np.uint8)
        plt.imshow(label)
        plt.show()
        # label = Image.fromarray(label)
        # label = self.transform_label(label)
        label = imresize(label, self.size)
        return img, label

    def __len__(self):
        return len(self.img_names)


def get_dataloader(args):
    transform_img = T.Compose([
        T.Resize(args["size"]),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    transform_label = T.Compose([
        T.Resize(args["size"]),
        T.ToTensor()
    ])
    train_set = Dataset(args["root"]+"/train", args["size"], transform_img, transform_label)
    validation_set = Dataset(args["root"]+"/validate", args["size"], transform_img, transform_label)
    train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=0, drop_last=False)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, validation_loader


def train(args, model, criterion, flag="unet"):
    losses_train_epoch = []
    losses_train = []
    losses_val = []
    loss_val_min, acc_val_min = 10, 0
    accs_train = []
    accs_val = []
    lr, print_interval, c, max_epoch, decay_epoch\
        = args["lr"], args["print_interval"], args["class_num"], args["epoch"], args["decay_epoch"]
    model_dir = "./model/" + args["name"] + ".pth"
    info_dir = "./model/" + args["name"] + "_info.pkl"
    for epoch in range(1, max_epoch+1):
        model.train()
        if epoch in decay_epoch:
            lr = lr / 10
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        running_loss, data_size, correct = 0, 0, 0
        print("Epoch", epoch, ":")
        print("learning rate =", lr)
        for t, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.cuda(), labels.cuda().long()
            if flag == "pixelnet":
                model.set_train_flag(True)
                rand_ind = generate_rand_ind(labels.cpu(), n_class=c, n_samples=2048)
                # rand_ind = np.random.choice(args['size'][0] * args['size'][1], 2000, replace=False)
                model.set_rand_ind(rand_ind)
                labels = labels.view(labels.size(0), -1)[:, rand_ind]
            # rand_ind = generate_rand_ind(labels, n_class=c, n_samples=4096)
            # outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)[:, :, rand_ind]
            # labels = labels.view(labels.size(0), -1)[:, rand_ind]
            outputs = model(inputs)
            predictions = outputs.cpu().argmax(1)
            loss = criterion(outputs, labels)
            correct += predictions.eq(labels.cpu()).sum().item()
            if (t + 1) % print_interval == 0:
                print("t = %d, loss = %.4f" % (t + 1, loss.item()))
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item() * labels.size(0)
            losses_train.append(loss.item())
            data_size += labels.size(0)
        loss_train = running_loss / data_size
        losses_train_epoch.append(loss_train)
        acc_train = correct / (data_size*np.prod(labels.size()))
        accs_train.append(acc_train)
        print("train loss:", round(loss_train, 5), "train accuracy:", round(acc_train, 5))

        with torch.no_grad():
            model.eval()
            running_loss, data_size, correct = 0, 0, 0
            for t, (inputs, labels) in enumerate(tqdm(validation_loader)):
                inputs, labels = inputs.cuda(), labels.cuda().long()
                if flag == "pixelnet":
                    model.set_train_flag(False)
                    labels = labels.view(labels.size(0), -1)
                outputs = model(inputs)
                predictions = outputs.argmax(1)
                loss = criterion(outputs, labels)
                correct += predictions.eq(labels).sum().item()
                # print statistics
                running_loss += loss.item() * labels.size(0)
                losses_train.append(loss.item())
                data_size += labels.size(0)
            loss_val = running_loss / data_size
            losses_val.append(loss_val)
            acc_val = correct / (data_size*np.prod(labels.size()))
            accs_val.append(acc_val)
            print('validation loss:', round(loss_val, 5), 'validation accuracy:', round(acc_val, 5))

        if loss_val < 10  and args['save']: # loss_val_min
            torch.save(model, model_dir)
            print('validation loss improved from', loss_val_min, 'to', loss_val, '. Model Saved.')
            print('validation accuracy:', acc_val)
            loss_val_min = loss_val
        # if acc_val > acc_val_min and args['save']:
        #     torch.save(model, model_dir)
        #     print('validation accuracy improved from', acc_val_min, 'to', acc_val, '. Model Saved.')
        #     acc_val_min = acc_val
        info = {'losses_train_epoch': losses_train_epoch,
                'losses_train': losses_train, 'losses_val': losses_val,
                'accs_train': accs_train, 'accs_val': accs_val}
        if args['save']:
            pickle.dump(info, open(info_dir, "wb"))


def validate(args, criterion, flag='unet'):
    model = torch.load('./model/' + args['name'] + '.pth')
    with torch.no_grad():
        model.eval()
        running_loss, data_size, correct = 0, 0, 0
        for t, (inputs, labels) in enumerate(tqdm(validation_loader)):
            inputs, labels = inputs.cuda(), labels.cuda().long()
            if flag == 'pixelnet':
                model.set_train_flag(False)
                labels = labels.view(labels.size(0), -1)
            outputs = model(inputs)
            predictions = outputs.argmax(1)
            loss = criterion(outputs, labels)
            correct += predictions.eq(labels).sum().item()
            # print statistics
            running_loss += loss.item() * labels.size(0)
            data_size += labels.size(0)
        loss_val = running_loss / data_size
        acc_val = correct / (data_size*np.prod(labels.size()))
        print('validation loss:', round(loss_val, 5), 'validation accuracy:', round(acc_val, 5))


if __name__ == "__main__":
    mode = "train" # train, test
    dataset = "tomography" # uhcs, ECCI, ECCI_small, ECCI_3, ECCI_small_processed, ECCI_3_processed, ECCI_small_processed
    method = "pixelnet" # pixelnet, unet, wnet
    version = "v1"

    if mode == 'train':
        criterion = nn.CrossEntropyLoss().cuda()
        args = get_config(dataset, version)
        if method == 'unet':
            model = unet_full(K=args['class_num']).cuda()
        elif method == 'pixelnet':
            model = pixelnet(K=args['class_num']).cuda()
        else:
            raise Exception("model cannot find")
        train_loader, validation_loader = get_dataloader(args)
        train(args, model, criterion, flag=method)

    elif mode == 'validate':
        criterion = nn.CrossEntropyLoss().cuda()
        args = get_config(dataset, version)
        _, validation_loader = get_dataloader(args)
        validate(args, criterion, flag=method)

    elif mode == 'test':
        args = get_config(dataset, version)
        imgs_dir = args["imgRoot"] + "/test/images/"
        img_names = os.listdir(imgs_dir)
        for name in img_names:
            img_dir = os.path.join(imgs_dir, name)
            save_dir = args["imgRoot"] + "/test/predictions/" + name[:-4] + "_" + method + name[-4:]
            prob, _ = get_prob(args, img_dir, method=method)
            visualize_prob(prob, save_dir=save_dir)

    else:
        args = get_config(dataset, version)

        # args = config.get_pixelnet_config(3)
        # visualize(args, 'uhcs0295.tif', pixel=True, save=True)

        # imgName = '9.tif'
        # prob, img_rgb = get_prob(args, imgName)
        # prediction_crf = process_crf(prob, img_rgb)
        # vis(prob, prediction_crf)
        # visualize_ECCI(args, imgName, save=False, npy=True)

        imgName = 'abd1-0.1-interface-6.tif'
        imgName = 'interior1.tif'
        imgName = '9.tif'
        imgName = 'abd1-0.1-8.tif'
        imgDir = args['imgRoot']+'/test/' + imgName
        save_dir = args['imgRoot']+'/test/' + imgName[:-4] + '_pred_' + dataset + '_' + method + '.tif'
        prob, _ = get_prob(args, imgDir, method=method)
        print(prob.shape)
        visualize_prob(prob, shape=(768, 1024), save_dir=save_dir)


