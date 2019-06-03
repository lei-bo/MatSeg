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

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from PIL import Image
from scipy.misc import imresize
from skimage.transform import resize
from tqdm import tqdm

import config
from main import *
import time

def conf(model_name, pixel=False):
    c = args['class_num']
    model = torch.load('./model/' + model_name + '.pth')
    with torch.no_grad():
        model.eval()
        conf_train, conf_val = meter.ConfusionMeter(c), meter.ConfusionMeter(c)
        # for t, (inputs, labels) in enumerate(tqdm(train_loader)):
        #     inputs, labels = inputs.cuda(), labels.cuda().long()
        #     if pixel:
        #         model.set_train_flag(False)
        #         labels = labels.view(labels.size(0), -1)
        #     outputs = model(inputs)
        #     conf_train.add(outputs.detach().permute(0,2,1).contiguous().view(-1,c), labels.detach().view(-1))
        ti=0
        for t, (inputs, labels) in enumerate(tqdm(validation_loader)):
            inputs, labels = inputs.cuda(), labels.cuda().long()
            if pixel:
                model.set_train_flag(False)
                labels = labels.view(labels.size(0), -1)
            start = time.time()
            outputs = model(inputs)
            end = time.time()
            ti += (end-start)/4
            if pixel:
                conf_val.add(outputs.detach().permute(0,2,1).contiguous().view(-1, c), labels.detach().view(-1))
            else:
                conf_val.add(outputs.detach().permute(0,2,3,1).contiguous().view(-1,c), labels.detach().view(-1))
        print(ti)
    return (conf_train, conf_val)

def evaluate(conf_mat):
    c = conf_mat.shape[0]
    precision = conf_mat.diagonal()/conf_mat.sum(0)
    recall = conf_mat.diagonal()/conf_mat.sum(1)
    IoUs = np.zeros(c)
    union_sum = 0
    for i in range(c):
        union = conf_mat[i,:].sum()+conf_mat[:,i].sum()-conf_mat[i,i]
        union_sum += union
        IoUs[i] = conf_mat[i,i]/union
    accuracy = conf_mat.diagonal().sum()/conf_mat.sum()
    IoU = IoUs.mean()
    print('accuracy:', accuracy, 'mIU:', IoU)
    print('precision:', precision)
    print('recall:', recall)
    print('IoUs:', IoUs)
    return accuracy, IoU, precision, recall, IoUs

def get_mean_std(var):
    var = np.array(var)
    if len(var.shape) == 1:
        return var.mean(), var.std()
    else:
        return var.mean(0), var.std(0)

def generate_rand_ind(labels, n_class, n_samples):
    n_samples_avg = int(n_samples/n_class)
    rand_ind = []
    for i in range(n_class):
        positions = np.where(labels.view(1, -1) == i)[1]
        if positions.size == 0:
            continue
        else:
            rand_ind.append(np.random.choice(positions, n_samples_avg))
    rand_ind = np.random.permutation(np.hstack(rand_ind))
    return rand_ind

def get_transformation(size):
    'size is a tuple (height * width)'
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    return transform

def visualize(args, imgName, pixel=False, save=False):
    imgDir = os.path.join(args['imgRoot'], imgName)
    model = torch.load('./model/' + args['name'] + '.pth')
    labels = args['labels_dic'][imgName[:-4]]
    transform = get_transformation(args['size'])
    with open(imgDir, 'rb') as f:
        img = Image.open(f)
        img = img.crop((0,0,645,484))
        if save:
            img.save('./result/paper/result/' + imgName[:-4] + '.tif')
        img = img.convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        if pixel:
            model.set_train_flag(False)
        inputs = img.cuda()
        outputs = model(inputs)
        predictions = outputs.argmax(1).squeeze().cpu().numpy()
    predictions = predictions.reshape(img.size(2),img.size(3))
    if save:
        pickle.dump(predictions, open('./result/paper/result/' + imgName[:-4] + '_label.pkl', "wb"))
        predictions = imresize(predictions, (484, 645), interp='nearest')
        pred_img = Image.fromarray(predictions)
        pred_img.save('./result/paper/result/' + imgName[:-4] + '_label.tif')
    fig = plt.figure()
    fig.set_size_inches(6.45,4.84)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(predictions)
    fig = plt.figure()
    fig.set_size_inches(6.45,4.84)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(labels)
    ax.axis('off')
    plt.show()

def visualize_wnet(imgName):
    imgDir = os.path.join(args['imgRoot'], imgName)
    model = torch.load('./model/' + args['name'] + '.pth')
    labels = args['labels_dic'][imgName[:-4]]
    transform = get_transformation(args['size'])
    with open(imgDir, 'rb') as f:
        img = Image.open(f)
        img = img.crop((0,0,645,484))
        img = img.convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input = img.cuda()
        output_img, output_seg = model(input)
        predictions = output_seg.argmax(1).squeeze().cpu().numpy()
        output_img = output_img.mean(1).squeeze().cpu().numpy()
    predictions = imresize(predictions.reshape(img.size(2),img.size(3)), (484,645), interp='nearest')
    fig = plt.figure()
    fig.set_size_inches(6.45,4.84)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(predictions)
    fig = plt.figure()
    fig.set_size_inches(6.45,4.84)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(output_img)
    fig = plt.figure()
    fig.set_size_inches(6.45,4.84)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(labels)
    ax.axis('off')
    plt.show()

def visualize_ECCI(args, imgName, pixel=False, save=False, npy=False):
    imgDir = os.path.join(args['imgRoot']+'/validate/images', imgName)
    model = torch.load('./model/' + args['name'] + '.pth')
    label_dir = args['imgRoot']+'/validate/labels/'+ imgName
    transform = get_transformation(args['size'])
    save_dir = './result/ECCI_result/'
    if npy:
        label = np.load(label_dir[:-4]+'.npy')
    else:
        label = plt.imread(label_dir)
    with open(imgDir, 'rb') as f:
        img = Image.open(f)
        if save:
            img.save(save_dir + imgName[:-4]+'.png')
        img = img.convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        if pixel:
            model.set_train_flag(False)
        inputs = img.cuda()
        outputs = model(inputs)
        predictions = outputs.argmax(1).squeeze().cpu().numpy()
    predictions = imresize(predictions.reshape(img.size(2),img.size(3)), (768,1024), interp='nearest')
    label = imresize(label.astype(int), (768, 1024), interp='nearest')
    if save:
        pred_img = Image.fromarray(predictions)
        pred_img.save(save_dir + imgName[:-4]+'_pred.png')
        label_save = Image.fromarray(label)
        label_save.save(save_dir + imgName[:-4]+'_label.png')
    fig = plt.figure()
    fig.set_size_inches(10.24,7.68)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(predictions)
    fig = plt.figure()
    fig.set_size_inches(10.24,7.68)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(label)
    ax.axis('off')
    plt.show()

def get_prob(args, imgDir, method='unet'):
    model = torch.load('./model/' + args['name'] + '.pth')
    transform = get_transformation(args['size'])
    save_dir = './result/ECCI_result/'
    with open(imgDir, 'rb') as f:
        img = Image.open(f)
        img_rgb = img.convert('RGB')
        img = transform(img_rgb)
        img = img.unsqueeze(0)
        img_rgb = np.array(img_rgb)
        img_rgb = imresize(img_rgb, args['size'], interp='nearest')
    with torch.no_grad():
        model.eval()
        if method == 'pixelnet':
            model.set_train_flag(False)
        inputs = img.cuda()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
    outputs = outputs.squeeze().cpu().numpy()
    outputs = outputs.reshape((outputs.shape[0], args['size'][0], args['size'][1]))
    return outputs, img_rgb

def process_crf(prob, img_rgb):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    d = dcrf.DenseCRF2D(640, 480, 3)
    unary = unary_from_softmax(prob)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=5, compat=3)
    d.addPairwiseBilateral(sxy=5, srgb=13, rgbim=img_rgb, compat=10)
    Q = d.inference(1)
    map = np.argmax(Q, axis=0).reshape((480, 640))
    return map

def vis(prob, pred_crf):
    pred = prob.argmax(0)
    pred = imresize(pred, (768, 1024), interp='nearest')
    pred_crf = imresize(pred_crf, (768, 1024), interp='nearest')
    fig = plt.figure()
    fig.set_size_inches(10.24,7.68)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(pred)
    fig = plt.figure()
    fig.set_size_inches(10.24,7.68)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pred_crf)
    ax.axis('off')
    plt.show()

def visualize_prob(prob, save_dir):
    pred = prob.argmax(0)
    # pred = imresize(pred, prob.shape, interp='nearest')
    if save_dir:
        # pred_img = Image.fromarray(pred)
        # pred_img.save(save_dir)
        plt.imsave(save_dir, pred, cmap='gray')
    # visualize_base(pred, pred.shape)

def visualize_base(pred, shape):
    fig = plt.figure()
    fig.set_size_inches(shape[1]/100, shape[0]/100)
    # fig.set_size_inches(10.24, 7.68)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(pred)
    plt.show()

if __name__ == '__main__':
    # args = config.unet_raw
    # args = config.get_unet_config(3)
    # train_loader, validation_loader = get_dataloader(args)
    # conf_mat = conf(args['name'], pixel=False)
    # conf_mat_val = conf_mat[1].value()
    # print(conf_mat_val.sum())
    # evaluate(conf_mat_val)
    # visualize('uhcs0295.tif', pixel=False)
    args = config.unet_ECCI
    # args = config.pixelnet_ECCI
    # visualize_ECCI('47.tif') # 150


    for i, name in enumerate([8,18,27,37,47,57,67,77,86,96,106,116,120,126,136,141,150]):
        visualize_ECCI(str(name)+'.tif', n=i)
    # train_loader, validation_loader = get_dataloader(args, flag='ECCI')
    # conf_mat = conf(args['name'], pixel=False)
    # conf_mat_val = conf_mat[1].value()
    # accuracy, IoU, precision, recall, IoUs = evaluate(conf_mat_val)
    # print(accuracy, IoU, precision, recall, IoUs)

    # accuracy_all, IoU_all, precision_all, recall_all, IoUs_all = [], [], [], [], []
    # for i in range(6):
    #     args = config.get_unet_config(i)
    #     train_loader, validation_loader = get_dataloader(args)
    #     conf_mat = conf(args['name'], pixel=False)
    #     conf_mat_val = conf_mat[1].value()
    #     # print(conf_mat_val.sum())
    #     accuracy, IoU, precision, recall, IoUs = evaluate(conf_mat_val)
    #     accuracy_all.append(accuracy)
    #     IoU_all.append(IoU)
    #     precision_all.append(precision)
    #     recall_all.append(recall)
    #     IoUs_all.append(IoUs)
    # print('accuracy:', get_mean_std(accuracy_all), 'mIU:', get_mean_std(IoU_all))
    # print('precision:', get_mean_std(precision_all))
    # print('recall:', get_mean_std(recall_all))
    # print('IoUs:', get_mean_std(IoUs_all))