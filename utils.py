import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    @property
    def average(self):
        return np.round(self.avg, 5)


class ScoreMeter:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, pred, label):
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                self.confusion_matrix[j][i] += np.logical_and(pred==i, label==j).sum()

    def get_scores(self, verbose=False):
        eps = 1e-8
        cm = self.confusion_matrix
        precision = cm.diagonal() / (cm.sum(axis=0) + eps)
        recall = cm.diagonal() / (cm.sum(axis=1) + eps)
        fraction_error = (cm.sum(axis=0) - cm.sum(axis=1)) / cm.sum()
        iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal() + eps)
        acc = cm.diagonal().sum() / cm.sum()
        miou = iou.mean()
        score_dict = {
            'accuracy': acc,
            'mIoU': miou,
            'IoUs': iou,
            'precision': precision,
            'recall': recall,
            'fraction_error': fraction_error
        }
        if verbose:
            print('\n'.join(f"{k}: {v:.5f}") for k, v in score_dict.items())
        return score_dict


class Recorder(object):
    def __init__(self, headers):
        self.headers = headers
        self.record = {}
        for header in self.headers:
            self.record[header] = []

    def update(self, vals):
        for header, val in zip(self.headers, vals):
            self.record[header].append(val)

    def save(self, path):
        pd.DataFrame(self.record).to_csv(path, index=False)


class ModelSaver:
    """A helper class to save the model with the best validation miou"""
    def __init__(self, model_path, delta=0):
        """
        :param model_path: the path to save the model to
        :param delta: minimum change in the monitored quantity to qualify as an
        improvement, defaults to 0
        """
        self.model_path = model_path
        self.best_epoch = 0
        self.best_score = np.NINF
        self.delta = delta

    def save_models(self, score, epoch, model, ious):
        if score > self.best_score + self.delta:
            print(f"validation iou improved from {self.best_score:.5f} to {score:.5f}.")
            self.best_score = score
            self.best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ious': ious
            }, self.model_path)


class LRScheduler:
    LR_SCHEDULER_MAP = {
        'CAWR': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'CyclicLR': optim.lr_scheduler.CyclicLR,
        'OneCycleLR': optim.lr_scheduler.OneCycleLR
    }
    STEP_EVERY_BATCH = ('CAWR', 'CyclicLR', 'OneCycleLR')

    def __init__(self, lr_scheduler_args, optimizer):
        args = lr_scheduler_args
        self.no_scheduler = False
        if args is None:
            self.no_scheduler = True
            return
        if args.type not in self.LR_SCHEDULER_MAP:
            raise ValueError(f"unsupported lr scheduler: {args.type}")
        else:
            self.lr_scheduler = self.LR_SCHEDULER_MAP[args.type](
                optimizer, **args.params
            )
        self.step_every_batch = args.type in self.STEP_EVERY_BATCH

    def step(self, last_batch=False):
        if self.no_scheduler:
            return
        if self.step_every_batch:
            self.lr_scheduler.step()
        else:
            if last_batch:
                self.lr_scheduler.step()


def get_optimizer(optimizer_args, model):
    args = optimizer_args
    list_params = [{'params': model.encoder.parameters(),
                    'lr': args.encoder_lr,
                    'weight_decay': args.weight_decay},
                   {'params': model.decoder.parameters(),
                    'lr': args.decoder_lr,
                    'weight_decay': args.weight_decay}]
    if args.type == 'Adam':
        optimizer = optim.Adam(list_params)
    elif args.type == 'AdamW':
        optimizer = optim.AdamW(list_params)
    elif args.type == 'SGD':
        optimizer = optim.SGD(list_params)
    else:
        raise ValueError(f"unsupported optimizer: {args.type}")
    return optimizer


class DiceLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target, eps=1e-8):
        """
        output logits shape is (N,C,H,W), target is (N,H,W)
        """
        batch_size, n_classes = output.shape[0], output.shape[1]
        pred = F.softmax(output, dim=1).view(batch_size, n_classes, -1)
        target = target.view(batch_size, -1)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred * mask.unsqueeze(1)
            target = target * mask
            target = F.one_hot((target * mask).to(torch.long), n_classes)  # N,H*W -> N,H*W,C
            target = target.permute(0, 2, 1) * mask.unsqueeze(1)  # H,C,H*W
        else:
            target = F.one_hot(target, n_classes)
            target = target.permute(0, 2, 1)
        inter = (pred * target).sum(dim=[0, 2])
        total = (pred + target).sum(dim=[0, 2])
        dice = 2. * inter / (total + eps)
        return 1. - dice.mean()


class JaccardLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target, eps=1e-8):
        """
        output logits shape is (N,C,H,W), target is (N,H,W)
        """
        batch_size, n_classes = output.shape[0], output.shape[1]
        pred = F.softmax(output, dim=1).view(batch_size, n_classes, -1)
        target = target.view(batch_size, -1)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred * mask.unsqueeze(1)
            target = target * mask
            target = F.one_hot((target * mask).to(torch.long), n_classes)  # N,H*W -> N,H*W,C
            target = target.permute(0, 2, 1) * mask.unsqueeze(1)  # H,C,H*W
        else:
            target = F.one_hot(target, n_classes)
            target = target.permute(0, 2, 1)
        inter = (pred * target).sum(dim=[0, 2])
        union = (pred + target).sum(dim=[0, 2]) - inter
        jaccard = inter / (union + eps)
        return 1. - jaccard.mean()


def get_loss_fn(loss_type, ignore_index):
    if loss_type == 'CE':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'Dice':
        return DiceLoss(ignore_index=ignore_index)
    elif loss_type == 'Jaccard':
        return JaccardLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"unsupported loss type: {loss_type}")
