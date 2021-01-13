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


class Recorder(object):
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)


def accuracy(predictions, labels):
    correct = predictions.eq(labels.cpu()).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc


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

