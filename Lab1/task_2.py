#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl
import time 
# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, iou
from PIL import Image, ImageDraw

from torchvision.ops import box_iou

# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# WANDB
USE_WANDB = False
def calculate_map(detections, gt_cnts):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters
    aps = []
    for class_num in range(20):
        if len(detections[class_num]) == 0:
            aps.append(0.0); continue
        dets = np.array(detections[class_num].sort(reverse=True))
        pfx = np.cumsum(dets, axis=0)
        precisions = pfx/(np.arange(len(pfx))+1)
        recalls = pfx/gt_cnts[class_num]
        precisions_desc = np.maximum.accumulate(precisions[::-1])[::-1]
        ap = 0.0
        # mAP@[0.05:0.95] COCO
        # reference
        # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
        for recall_tick in np.arange(0.05, 0.95, 0.05):
            i = 0
            while recalls[i] < recall_tick: i += 1
            ap += precisions_desc[i]
        aps.append(ap)
    return aps

def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    # for each class store (pred_score, T/F)
    detections = [[] for i in range(20)]
    # store how many instances of each class exists in valid set.(gt)
    gt_cnts = [0] * 20
    end_time = time.time()
    
    viz_images, viz_boxes = [], []
    viz_cnt = 0
    print("evaluating.. ", len(val_loader))
    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO (Q2.3): perform forward pass, compute cls_probs
            n_rois = len(rois)
            h, w = image.shape[-2:]
            scale_factor = torch.Tensor([w, h, w, h])
            roi_pix = [
                torch.cat(bbox)*scale_factor for bbox in rois
            ]
            roi_pix  = torch.stack(roi_pix, dim=0).float().cuda()
            cls_probs = model.forward(image, roi_pix, target)
            
            nms_bboxes = [] # store bbox with class_num, for this image
            # TODO (Q2.3): Iterate over each class (follow comments)
            for cl in gt_class_list:
                gt_cnts[cl.item()] += 1
                
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                cls_scores = cls_probs[:, class_num]
                used = [False] * len(gt_boxes)
                for idx, pred_bbox in enumerate(rois):
                    # filter low scores (negatives)
                    if cls_scores[idx] < thresh: continue
                    # iterate over gt_bboxes and find matching bbox
                    found = False
                    for gt_idx, gt_bbox in enumerate(gt_boxes):
                        if gt_class_list[gt_idx] != class_num: continue
                        if not used[gt_idx] and box_iou(torch.cat(pred_bbox).unsqueeze(0), torch.cat(gt_bbox).unsqueeze(0)).item() > 0.5:
                            # a true positive
                            detections[class_num].append((cls_scores[idx].item(), 1))
                            used[gt_idx] = True; found=True 
                    
                    # if no matching gt_bbox found for this prediction, it's a FP
                    if not found:
                        detections[class_num].append((cls_scores[idx].item(), 0))
                            
                # use NMS to get boxes and scores for vis
                if viz_cnt < 10: # first ten images
                    bboxes, scores = nms(roi_pix, cls_scores, 0.3)
                    nms_bboxes += [{
                                "position": {
                                    "minX": bbox[0],
                                    "minY": bbox[1], #top(min row)
                                    "maxX": bbox[2],
                                    "maxY": bbox[3], #bottom(max row)
                                    },
                                "class_id": class_num
                            } for bbox in bboxes]

            # draw random 10 images with bounding boxes
            USE_WANDB= True #debugging.. remove this 
            if USE_WANDB and viz_cnt < 10 and len(nms_bboxes) > 0:
                viz_images.append(tensor_to_PIL(data['image']))
                viz_boxes.append({
                        "proposals": {
                            "box_data": nms_bboxes
                        },
                    })
                viz_cnt += 1
            USE_WANDB = False #debugging.. remove this
                
        print(f"eval_time: {time.time()-end_time}")
        ap = calculate_map(detections, gt_cnts)
        # TODO (Q2.3): visualize bounding box predictions when required          
        if USE_WANDB:
            wdb_img = [wandb.Image(img, boxes=box) for img, box in zip(viz_images, viz_boxes)]
            wandb.log({"region proposals": wdb_img})
    return ap
        


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    for epoch in range(args.epochs):
        end = time.time()
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            
            h, w = image.shape[-2:]
            scale_factor = torch.Tensor([w, h, w, h])
            rois = [
                torch.cat(bbox)*scale_factor for bbox in rois
            ]
            rois  = torch.stack(rois, dim=0).float().cuda()
            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            model.forward(image, rois, target)

            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # average loss
            losses.update(loss.item())
            
            # measure average elapsed time per each forward
            batch_time.update(time.time() - end)
            end = time.time()
            
            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if iter % args.val_interval == 0 and iter != 0:
                # testing section
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", ap)
                if USE_WANDB:
                    # log mAP and per class AP
                    dic = {f"eval/ap_{cls_id}":cls_ap for cls_id, cls_ap in enumerate(ap)}
                    dic.update({"eval/mAP": np.mean(ap)})
                    wandb.log(dic)
                model.train()
                # logging section
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch,
                      iter,
                      len(train_loader),
                      batch_time=batch_time,
                      loss=losses))
                if USE_WANDB:
                    wandb.log({'iteration': step_cnt, "train/loss": losses.avg})
                    

        # TODO (Q2.4): Perform all visualizations here
        # The intervals for different things are defined in the handout

                
    

def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    global USE_WANDB
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    if USE_WANDB:
        wandb.init(project="vlr-hw1")
        
    train_dataset = VOCDataset('trainval', image_size=512, top_n=300)
    val_dataset = VOCDataset('test', image_size=512, top_n=300)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        #print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            #print('Copied {}'.format(name))
        except:
            #print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for name, param in net.named_parameters():
        if name in pret_net:
            param.requires_grad = False
            print("froze ", name)
        elif 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
            print("init ", name)

    
            
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum)

    # Training
    train_model(net, train_loader, val_loader, optimizer, args)


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


if __name__ == '__main__':
    main()


