import numpy as np


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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


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

