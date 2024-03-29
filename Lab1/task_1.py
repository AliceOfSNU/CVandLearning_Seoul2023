import argparse
import os
import shutil
import time

import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

import cv2

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *

USE_WANDB = False  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_ID_TO_LABEL = {}
'''
returns the sum of k binary logistic regression losses
log(1+exp(-yk*xk)), where yk={-1, 1}
Arguments:
    x{torch.Tensor}: *k logit prediction values
    target{torch.Tensor}: *k. 1 if class exists in image, 0 otherwise
'''
def k_binary_logit_loss(x, target):
    return torch.log(1+torch.exp(-(target * 2 - 1)*x)).mean()
    
def main():
    global args, best_prec1, CLASS_NAMES, CLASS_ID_TO_LABEL
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    args.pretrained = True
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    #model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    criterion = k_binary_logit_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    train_dataset = VOCDataset('trainval', top_n=10)
    val_dataset = VOCDataset('test', top_n=10)
    CLASS_ID_TO_LABEL = dict(enumerate(CLASS_NAMES))
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn = VOC_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn = VOC_collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    USE_WANDB = True
    if USE_WANDB:
        wandb.init(project="vlr-hw1")
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data['label'].cuda()
        image = data['image'].cuda()

        # TODO (Q1.1): Get output from model
        imoutput = model.forward(image) #256, 20, 11, 11

        # TODO (Q1.1): Perform any necessary operations on the output
        feature_size = imoutput.shape[-1]
        preds = F.max_pool2d(imoutput, feature_size).squeeze()#256, 20, 1
        
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(preds, target)

        # measure metrics and record loss
        m1 = metric1(preds.cpu().detach().numpy(), target.cpu().detach().numpy())
        m2 = metric2(preds.cpu().detach().numpy(), target.cpu().detach().numpy())
        losses.update(loss.item())
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals
        wandb.log({'iteration': epoch*iters_per_epoch+i, 'train/loss': loss.item(), 'train/m1': m1, 'train/m2': m2})
        if (epoch == 1 or epoch % 15 == 0) and i == 2:
            imgs = []; heatmaps = []  
            for idx in range(2):
                img = image[idx].cpu().detach()
                gt_cls = data['gt_classes'][idx][0]
                imgs.append(wandb.Image(tensor_to_PIL(img),
                        boxes={
                            "predictions": {
                                "box_data": get_box_data([gt_cls], [data['gt_boxes'][idx][0]]),
                                "class_labels": CLASS_ID_TO_LABEL,
                            },
                        }))
                # create heatmap
                heatmap = imoutput[idx][gt_cls].cpu().detach().numpy()
                heatmap = heatmap-np.min(heatmap)
                heatmap /= np.max(heatmap)
                heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmaps.append(wandb.Image(heatmap))
            wandb.log({'input_img': imgs, 'heatmaps': heatmaps})
        # End of train()
        


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data['label'].cuda()
        image = data['image'].cuda()

        # TODO (Q1.1): Get output from model
        imoutput = model.forward(image) #256, 20, 11, 11

        # TODO (Q1.1): Perform any necessary functions on the output
        feature_size = imoutput.shape[-1]
        preds = F.max_pool2d(imoutput, feature_size).squeeze()#256, 20, 1
        
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(preds, target)

        # measure metrics and record loss
        m1 = metric1(preds.cpu().detach().numpy(), target.cpu().detach().numpy())
        m2 = metric2(preds.cpu().detach().numpy(), target.cpu().detach().numpy())
        losses.update(loss.item())
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q 1.3): Visualize things as mentioned in handout
        ''' UNCOMMENT FOR Q!.3is more tuned to this dataset. Even though there is a steep 
        
        # TODO (Q1.3): Visualize at appropriate intervals
        if i == 1:
            imgs = []; heatmaps = []  
            for idx in range(4, 8):
                img = image[idx].cpu().detach()
                gt_cls = data['gt_classes'][idx][0]
                imgs.append(wandb.Image(tensor_to_PIL(img),
                        boxes={
                            "predictions": {
                                "box_data": get_box_data([gt_cls], [data['gt_boxes'][idx][0]]),
                                "class_labels": CLASS_ID_TO_LABEL,
                            },
                        }))
                # create heatmap
                heatmap = imoutput[idx][gt_cls].cpu().detach().numpy()
                heatmap = heatmap-np.min(heatmap)
                heatmap /= np.max(heatmap)
                heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmaps.append(wandb.Image(heatmap))
            wandb.log({'input_img': imgs, 'heatmaps': heatmaps})
        
        '''
    wandb.log({'epoch':epoch, 'valid/loss': loss.item(), 'valid/m1': avg_m1.avg, 'valid/m2':avg_m2.avg}) 
    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def metric1(output, target):
    # TODO (Q1.5): compute metric1
    num_classes = output.shape[-1]
    output = 1/(1+np.exp(-output))
    ap = 0.0; valid_cls = 0
    for i in range(num_classes):
        if target[:, i].sum() == 0: continue #no positive class in batch
        valid_cls += 1
        ap += sklearn.metrics.average_precision_score(target[:, i], output[:, i])
    return ap/valid_cls


def metric2(output, target):
    output = 1/(1+np.exp(-output))
    output = np.where(output>0.6, 1, 0)
    recall = sklearn.metrics.recall_score(target, output, average='micro')
    return recall


if __name__ == '__main__':
    main()
