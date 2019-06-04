import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader

import torchvision.transforms as T

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils import AverageMeter, generate_rand_ind, accuracy
from models import pixelnet, unet
from config import get_config


class Dataset(data.Dataset):
    def __init__(self, root, size, transform_img, transform_label):
        self.img_root = root + '/images/'
        self.label_root = root + '/labels_npy/'
        self.img_names = os.listdir(self.img_root)
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_dir = self.img_root + img_name
        label_dir = '%s%s.npy' % (self.label_root, img_name[:-4])  # the name of the label numpy file
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform_img(img)
        label = np.load(label_dir).astype(np.int8)
        label = Image.fromarray(label)
        label = self.transform_label(label).squeeze()
        return img, label

    def __len__(self):
        return len(self.img_names)


def get_dataloader(args):
    transform_img = T.Compose([
        T.Resize(args['size']),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    transform_label = T.Compose([
        T.Resize(args['size']),
        T.ToTensor()
    ])
    train_set = Dataset(args['root']+'/train', args['size'], transform_img, transform_label)
    validation_set = Dataset(args['root']+'/validate', args['size'], transform_img, transform_label)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=False)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, validation_loader


def train(args, model, criterion, optimizer, train_loader, flag='unet'):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')
    model.train()
    print('learning rate =', optimizer.param_groups[0]['lr']) # get learning rate from optimizer status
    for t, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.cuda(), labels.cuda().long()
        if flag == 'pixelnet':
            model.set_train_flag(True)
            rand_ind = generate_rand_ind(labels.cpu(), n_class=args['n_class'], n_samples=2048)
            model.set_rand_ind(rand_ind)
            labels = labels.view(labels.size(0), -1)[:, rand_ind]

        # compute output
        outputs = model(inputs)
        predictions = outputs.cpu().argmax(1)
        loss = criterion(outputs, labels)

        # measure accuracy, record loss
        accs.update(accuracy(predictions, labels), inputs.shape[0])
        losses.update(loss.item(), inputs.size(0))

        if (t + 1) % args['print_freq'] == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.item()))

        # compute gradient and do gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('train loss:', round(losses.avg, 5), 'train accuracy:', round(accs.avg, 5))


def validate(model, criterion, validation_loader, flag='unet'):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')
    with torch.no_grad():
        model.eval()
        for t, (inputs, labels) in enumerate(tqdm(validation_loader)):
            inputs, labels = inputs.cuda(), labels.cuda().long()
            if flag == 'pixelnet':
                model.set_train_flag(False)
                labels = labels.view(labels.size(0), -1)

            # compute output
            outputs = model(inputs)
            predictions = outputs.cpu().argmax(1)
            loss = criterion(outputs, labels)

            # measure accuracy, record loss
            accs.update(accuracy(predictions, labels), inputs.shape[0])
            losses.update(loss.item(), inputs.size(0))
        print('validation loss:', round(losses.avg, 5), 'validation accuracy:', round(accs.avg, 5))
    return losses.avg, accs.avg


def main(mode, dataset, version, save):
    args = get_config(dataset, version)
    method = args['model']
    if mode == 'train':
        criterion = nn.CrossEntropyLoss().cuda()
        if method == 'unet':
            model = unet(K=args['n_class']).cuda()
        elif method == 'pixelnet':
            model = pixelnet(K=args['n_class']).cuda()
        else:
            raise Exception('model cannot find')

        loss_val_min, acc_val_min = 10, 0
        model_dir = './saved/%s_%s.pth' % (args['name'], method)
        # info_dir = './saved/' + args['name'] + '_info.pkl'
        train_loader, validation_loader = get_dataloader(args)
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        for epoch in range(1, args['epoch'] + 1):
            print('Epoch', epoch, ':')
            train(args, model, criterion, optimizer, train_loader, flag=method)
            loss_val, acc_val = validate(model, criterion, validation_loader, flag=method)
            scheduler = ReduceLROnPlateau(optimizer, patience=5)
            scheduler.step(loss_val)

            # save model at lower loss_val
            if loss_val < loss_val_min and save:
                torch.save(model, model_dir)
                print('validation loss improved from', loss_val_min, 'to', loss_val, '. Model Saved.')
                loss_val_min = loss_val

            # save model at higher acc_val
            # if acc_val > acc_val_min and save:
            #     torch.save(model, model_dir)
            #     print('validation accuracy improved from', acc_val_min, 'to', acc_val, '. Model Saved.')
            #     acc_val_min = acc_val

    elif mode == 'validate':
        criterion = nn.CrossEntropyLoss().cuda()
        model_dir = './saved/%s_%s.pth' % (args['name'], method)
        model = torch.load(model_dir)
        _, validation_loader = get_dataloader(args)
        validate(model, criterion, validation_loader, flag=method)

    elif mode == 'test':
        imgs_dir = args['imgRoot'] + '/test/images/'
        img_names = os.listdir(imgs_dir)
        for name in img_names:
            img_dir = os.path.join(imgs_dir, name)
            save_dir = args['imgRoot'] + '/test/predictions/' + name[:-4] + '_' + method + name[-4:]
            prob, _ = get_prob(args, img_dir, method=method)
            visualize_prob(prob, save_dir=save_dir)
    else:
        raise Exception('mode cannot find')

if __name__ == '__main__':
    mode = 'train'  # train, validate, test
    # uhcs, ECCI, ECCI_small, ECCI_3, ECCI_small_processed, ECCI_3_processed, ECCI_small_processed
    dataset = 'tomography'
    version = 'v2'
    save = False
    main(mode, dataset, version, save)
