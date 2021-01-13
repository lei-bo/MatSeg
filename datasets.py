import os
import numpy as np
import random
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, root, size, transform_img, transform_label):
        self.img_root = root + '/images/'
        self.label_root = root + '/labels_npy/'
        self.img_names = os.listdir(self.img_root)
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.seeds = [np.random.randint(np.iinfo(np.int32).max) for _ in range(len(self.img_names))]

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_dir = self.img_root + img_name
        label_dir = '%s%s.npy' % (self.label_root, img_name[:-4])  # the name of the label numpy file
        seed = self.seeds[index] # get a random seed for fixed random transformation
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            random.seed(seed)
            img = self.transform_img(img)
        label = np.load(label_dir).astype(np.int8)
        label = Image.fromarray(label)
        random.seed(seed)
        label = self.transform_label(label).squeeze()
        return img, label, img_name

    def __len__(self):
        return len(self.img_names)


def get_transform(config, is_train):
    mean, std, is_aug = config['mean'], config['std'], config['aug']
    if is_train and is_aug:
        transform_label = T.Compose([
            T.RandomRotation(45),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])
    else:
        transform_label = T.Compose([
            T.ToTensor()
        ])

    transform_img = T.Compose([
        transform_label,
        T.Normalize(mean=mean, std=std)
    ])
    return transform_img, transform_label


def get_dataloader(config):
    transform_img_train, transform_label_train = get_transform(config, is_train=True)
    transform_img_val, transform_label_val = get_transform(config, is_train=False)
    train_set = Dataset(config['root']+'/train', config['size'], transform_img_train, transform_label_train)
    val_set = Dataset(config['root']+'/validate', config['size'], transform_img_val, transform_label_val)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, val_loader