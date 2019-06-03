from torch.utils import data
import torchvision.transforms as T

import os
from PIL import Image
from scipy.misc import imresize
import numpy as np
import pickle

class dset(data.Dataset):
    def __init__(self, root, labels, size, transform_img, transform_label, flag, validation_set):
        self.root = root
        imgNames = os.listdir(root)
        trainNames, valNames = self.split(imgNames, validation_set)
        if flag:
            self.imgNames = valNames
        else:
            self.imgNames = trainNames
        self.imgDirs = [os.path.join(root, name) for name in self.imgNames]
        self.labels = labels
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label
    def split(self, imgNames, validation_set):
        trainNames, valNames = [], []
        for name in imgNames:
            if name[-8:-4] in validation_set:
                valNames.append(name)
            else:
                trainNames.append(name)
        return trainNames, valNames
    def __getitem__(self, index):
        imgDir = self.imgDirs[index]
        imgName = self.imgNames[index]
        label = self.labels[imgName[:-4]]
#         label = np.expand_dims(label, axis=2)
        label = imresize(label, self.size)
        with open(imgDir, 'rb') as f:
            img = Image.open(f)
            img = img.crop((0,0,645,484))
            img = img.convert('RGB')
            img = self.transform_img(img)
        return img, label
    def __len__(self):
        return len(self.imgNames)

class dset_wnet(data.Dataset):
    def __init__(self, root, size, transform_img):
        self.root = root
        self.imgNames = os.listdir(root)
        self.imgDirs = [os.path.join(root, name) for name in self.imgNames]
        self.size = size
        self.transform_img = transform_img
    def __getitem__(self, index):
        imgDir = self.imgDirs[index]
        imgName = self.imgNames[index]
        with open(imgDir, 'rb') as f:
            img = Image.open(f)
            img = img.crop((0,0,645,484))
            img = img.convert('RGB')
            img_RGB = T.Resize(self.size)(img)
            img_RGB = T.functional.to_tensor(img_RGB)
            img = self.transform_img(img)
        feature = pickle.load(open('./data/uhcs/hypercolumn/'+imgName[:-4]+'.pkl', "rb"))
        return img, feature
    def __len__(self):
        return len(self.imgNames)

class dset_all(data.Dataset):
    def __init__(self, root, labels, size, transform_img):
        self.root = root
        self.imgNames = os.listdir(root)
        self.imgDirs = [os.path.join(root, name) for name in self.imgNames]
        self.labels = labels
        self.size = size
        self.transform_img = transform_img
    def __getitem__(self, index):
        imgDir = self.imgDirs[index]
        imgName = self.imgNames[index]
        if self.labels == False:
            label = np.zeros(self.size)
        else:
            label = self.labels[imgName[:-4]]
            # label = np.expand_dims(label, axis=2)
            label = imresize(label, self.size)
        with open(imgDir, 'rb') as f:
            img = Image.open(f)
            if 'uhcs' in self.root or 'spheroidite' in self.root:
                img = img.crop((0,0,645,484))
            img = img.convert('RGB')
            img = self.transform_img(img)
        return img, label, imgName
    def __len__(self):
        return len(self.imgNames)

class dset_ECCI(data.Dataset):
    def __init__(self, root, size, transform_img, transform_label):
        img_root, label_root = root + '/images', root + '/labels'
        self.img_names = os.listdir(img_root)
        self.img_dirs = [os.path.join(img_root, name) for name in self.img_names]
        self.label_names = os.listdir(label_root)
        self.label_dirs = [os.path.join(label_root, name) for name in self.label_names]
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label
    def __getitem__(self, index):
        img_dir = self.img_dirs[index]
        label_dir = self.label_dirs[index]
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform_img(img)
        with open(label_dir, 'rb') as f:
            label = Image.open(f)
            label = np.array(label).astype(np.uint8)
            label = imresize(label, self.size)
        return img, label
    def __len__(self):
        return len(self.img_names)

class dset_segmentation_npy(data.Dataset):
    def __init__(self, root, size, transform_img, transform_label):
        img_root, label_root = root + '/images', root + '/labels'
        self.img_names = os.listdir(img_root)
        self.img_dirs = [os.path.join(img_root, name) for name in self.img_names]
        self.label_names = os.listdir(label_root)
        self.label_dirs = [os.path.join(label_root, name) for name in self.label_names]
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label
    def __getitem__(self, index):
        img_dir = self.img_dirs[index]
        label_dir = self.label_dirs[index][:-4] + '.npy'
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform_img(img)

        label = np.load(label_dir).astype(np.uint8)
        label = imresize(label, self.size)
        return img, label
    def __len__(self):
        return len(self.img_names)
