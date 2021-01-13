import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import Dataset, get_dataloader, get_transform
from utils import AverageMeter, Recorder, accuracy, metrics
from models import model_mappings
from config import get_config


def train(config, model, criterion, optimizer, train_loader, method):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')
    model.train()
    print('learning rate =', optimizer.param_groups[0]['lr'])  # get learning rate from optimizer status
    for t, (inputs, labels, _) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.cuda(), labels.cuda().long()
        if method == 'pixelnet':
            model.set_train_flag(True)
            rand_ind = model.generate_rand_ind(labels.cpu(), n_class=config['n_class'], n_samples=2048)
            model.set_rand_ind(rand_ind)
            labels = labels.view(labels.size(0), -1)[:, rand_ind]

        # compute output
        outputs = model(inputs)
        predictions = outputs.cpu().argmax(1)
        loss = criterion(outputs, labels)

        # measure accuracy, record loss
        accs.update(accuracy(predictions, labels), inputs.shape[0])
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('--- training result ---')
    print('loss: %.5f, accuracy: %.5f' % (losses.avg, accs.avg))
    return losses.avg, accs.avg


def evaluate(config, model, criterion, validation_loader, method, test_flag=False, save_dir=None):
    losses = AverageMeter('Loss', ':.5f')
    conf_meter = ConfusionMeter(config['n_class'])
    with torch.no_grad():
        model.eval()
        for t, (inputs, labels, names) in enumerate(tqdm(validation_loader)):
            inputs, labels = inputs.cuda(), labels.cuda().long()
            if method == 'pixelnet':
                model.set_train_flag(False)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # save predictions if needed
            predictions = outputs.cpu().argmax(1)
            if test_flag:
                for i in range(predictions.shape[0]):
                    plt.imsave('%s/%s.png' % (save_dir, names[i][:-4]), predictions[i].squeeze(), cmap='gray')

            # measure accuracy, record loss
            losses.update(loss.item(), inputs.size(0))
            conf_meter.add(outputs.permute(0, 2, 3, 1).contiguous().view(-1, config['n_class']), labels.view(-1))
        if test_flag:
            print('--- evaluation result ---')
        else:
            print('--- validation result ---')
        conf_mat = conf_meter.value()
        acc, iou = metrics(conf_mat, verbose=test_flag)
        print('loss: %.5f, accuracy: %.5f, IU: %.5f' % (losses.avg, acc, iou))
    return losses.avg, acc, iou


def main(args):
    if args.seed:
        np.random.seed(int(args.seed))
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
    config = get_config(args.dataset, args.version)
    method = config['model']
    criterion = nn.CrossEntropyLoss().cuda()
    try:
        model = model_mappings[method](config['n_class']).cuda()
    except KeyError:
        print('%s model does not exist' % method)
        sys.exit(1)

    model_dir = './saved/%s_%s.pth' % (config['name'], method)
    if args.mode == 'train':
        log_dir = './log/%s_%s.log' % (config['name'], method)
        train_loader, validation_loader = get_dataloader(config)
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
        else:
            print('cannot found %s optimizer' % config['optimizer'])
            sys.exit(1)

        scheduler = ReduceLROnPlateau(optimizer, patience=3)
        recorder = Recorder(('loss_train', 'acc_train', 'loss_val', 'acc_val'))
        iou_val_max = 0
        for epoch in range(1, config['epoch'] + 1):
            print('Epoch %s:' % epoch)
            loss_train, acc_train = train(config, model, criterion, optimizer, train_loader, method=method)
            loss_val, acc_val, iou_val = evaluate(config, model, criterion, validation_loader, method=method)
            scheduler.step(loss_train)

            # update loss and accuracy per epoch
            recorder.update((loss_train, acc_train, loss_val, acc_val))
            if args.save: torch.save(recorder.record, log_dir)

            # save model with higher iou
            if iou_val > iou_val_max:
                print('validation iou improved from %.5f to %.5f.' % (iou_val_max, iou_val))
                iou_val_max = iou_val
                if args.save:
                    print('Model saved.')
                    torch.save({
                        'epoch': epoch,
                        'version': args.version,
                        'config': config,
                        'model_state_dict': model.state_dict(),
                    }, model_dir)

    elif args.mode == 'evaluate':
        test_dir = '%s/%s' % (config['root'], args.test_folder)
        test_set = Dataset(test_dir, config['size'], *get_transform(config, is_train=False))
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        model.load_state_dict(torch.load(model_dir)['model_state_dict'])

        # save prediction results, make directory if not exists
        save_dir = '%s/predictions/%s_%s' % (test_dir, args.version, method)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        evaluate(config, model, criterion, test_loader, method=method, test_flag=True, save_dir=save_dir)

    else:
        print('%s mode does not exist' % args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Micrograph Segmentation')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('mode', choices=['train', 'evaluate'],
                        help='mode choices: train, evaluate')
    parser.add_argument('version', help='version defined in config.py (v1, v2, ...)')
    parser.add_argument('--save', action='store_true', help='save the trained model')
    parser.add_argument('--test-folder', default='test', help='name of the folder running test')
    parser.add_argument('--seed', default=None, help='random seed to reproduce same results')
    args = parser.parse_args()
    main(args)
