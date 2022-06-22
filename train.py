from tqdm import tqdm

from args import Arguments
from UNet import UNetVgg16
from datasets import get_dataloaders
from eval import eval_epoch
from utils import AverageMeter, ScoreMeter, Recorder, ModelSaver, LRScheduler, get_optimizer, get_loss_fn


def train_epoch(model, dataloader, n_classes, optimizer, lr_scheduler, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    score_meter = ScoreMeter(n_classes)
    for i, (inputs, labels, _) in enumerate(tqdm(dataloader, ncols=0, leave=False)):
        inputs, labels = inputs.to(device), labels.long().to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.detach().cpu().numpy().argmax(axis=1)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(last_batch=(i == len(dataloader)-1))
        # measure
        loss_meter.update(loss.item(), inputs.size(0))
        score_meter.update(preds, labels.cpu().numpy())

    scores = score_meter.get_scores()
    miou, ious, acc = scores['mIoU'], scores['IoUs'], scores['accuracy']
    return loss_meter.avg, acc, miou, ious


def train(args):
    Arguments.save_args(args, args.args_path)
    train_loader, val_loader, _ = get_dataloaders(args)
    model = UNetVgg16(n_classes=args.n_classes).to(args.device)
    optimizer = get_optimizer(args.optimizer, model)
    lr_scheduler = LRScheduler(args.lr_scheduler, optimizer)
    criterion = get_loss_fn(args.loss_type, args.ignore_index).to(args.device)
    model_saver = ModelSaver(args.model_path)
    recorder = Recorder(['train_miou', 'train_acc', 'train_loss',
                         'val_miou', 'val_acc', 'val_loss'])
    for epoch in range(args.n_epochs):
        print(f"{args.experim_name} Epoch {epoch+1}:")
        train_loss, train_acc, train_miou, train_ious = train_epoch(
            model=model,
            dataloader=train_loader,
            n_classes=args.n_classes,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=args.device,
        )
        print(f"train | mIoU: {train_miou:.3f} | accuracy: {train_acc:.3f} | loss: {train_loss:.3f}")
        val_loss, val_scores = eval_epoch(
            model=model,
            dataloader=val_loader,
            n_classes=args.n_classes,
            criterion=criterion,
            device=args.device,
        )
        val_miou, val_ious, val_acc = val_scores['mIoU'], val_scores['IoUs'], val_scores['accuracy']
        print(f"valid | mIoU: {val_miou:.3f} | accuracy: {val_acc:.3f} | loss: {val_loss:.3f}")
        recorder.update([train_miou, train_acc, train_loss, val_miou, val_acc, val_loss])
        recorder.save(args.record_path)
        if args.metric.startswith("IoU"):
            metric = val_ious[int(args.metric.split('_')[1])]
        else: metric = val_miou
        model_saver.save_models(metric, epoch+1, model,
                                ious={'train': train_ious, 'val': val_ious})

    print(f"best model at epoch {model_saver.best_epoch} with miou {model_saver.best_score:.5f}")


if __name__ == '__main__':
    arg_parser = Arguments()
    args = arg_parser.parse_args(verbose=True)
    train(args)
