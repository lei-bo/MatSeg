import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from args import Arguments
from UNet import UNetVgg16
from datasets import get_dataloaders
from utils import AverageMeter, ScoreMeter, get_loss_fn


@torch.no_grad()
def eval_epoch(model, dataloader, n_classes, criterion, device, pred_dir=None):
    model.eval()
    loss_meter = AverageMeter()
    score_meter = ScoreMeter(n_classes)
    for inputs, labels, names in tqdm(dataloader, ncols=0, leave=False):
        inputs, labels = inputs.to(device), labels.long().to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.detach().cpu().numpy().argmax(axis=1)
        # measure
        loss_meter.update(loss.item(), inputs.size(0))
        score_meter.update(preds, labels.cpu().numpy())
        # save predicted results
        if pred_dir:
            assert preds.shape[0] == 1
            np.save(f"{pred_dir}/{names[0].split('.')[0]}.npy",
                    preds[0].astype(np.int8))

    scores = score_meter.get_scores()
    return loss_meter.avg, scores


def evaluate(args, mode, save_pred=False):
    _, val_loader, test_loader = get_dataloaders(args)
    if mode == 'val':
        dataloader = val_loader
    elif mode == 'test':
        dataloader = test_loader
    else:
        raise ValueError(f"{mode} not supported. Choose from 'val' or 'test'")
    model = UNetVgg16(n_classes=args.n_classes).to(args.device)
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'], strict=False)
    criterion = get_loss_fn(args.loss_type, args.ignore_index).to(args.device)
    eval_loss, scores = eval_epoch(
        model=model,
        dataloader=dataloader,
        n_classes=args.n_classes,
        criterion=criterion,
        device=args.device,
        pred_dir=save_pred and args.pred_dir
    )
    miou, acc = scores['mIoU'], scores['accuracy']
    print(f"{mode} | mIoU: {miou:.3f} | accuracy: {acc:.3f} | loss: {eval_loss:.3f}")
    return scores


if __name__ == '__main__':
    arg_parser = Arguments()
    arg_parser.parser.add_argument('--mode', '-m', choices=['val', 'test'],
                                   required=True)
    args = arg_parser.parse_args()
    evaluate(args, args.mode, save_pred=True)
