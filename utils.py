import numpy as np
import torch
import torchvision.transforms as T


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)


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


def accuracy(predictions, labels):
    correct = predictions.eq(labels.cpu()).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc


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


def metrics(conf_mat, verbose=True):
    c = conf_mat.shape[0]
    # ignore dividing by zero
    np.seterr(divide='ignore', invalid='ignore')
    precision = conf_mat.diagonal()/conf_mat.sum(0)
    recall = conf_mat.diagonal()/conf_mat.sum(1)
    IoUs = np.zeros(c)
    union_sum = 0
    for i in range(c):
        union = conf_mat[i,:].sum()+conf_mat[:,i].sum()-conf_mat[i,i]
        union_sum += union
        IoUs[i] = conf_mat[i,i]/union
    acc = conf_mat.diagonal().sum()/conf_mat.sum()
    if c == 2:
        IoU = IoUs[1]
    else:
        IoU = IoUs.mean()
    if verbose:
        print('precision:', np.round(precision, 5), precision.mean())
        print('recall:', np.round(recall, 5), recall.mean())
        print('IoUs:', np.round(IoUs, 5), IoUs.mean())
    return acc, IoU

