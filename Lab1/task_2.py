#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
import pickle as pkl
import time 
import tqdm

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
    default=6,
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
rand_seed = 12
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# GLOBAL
USE_WANDB = True
CLASS_NAMES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

def calculate_map(detections, gt_cnts):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # return: per-class mAP, area under precision/recall curve
    # Feel free to write necessary function parameters
    aps = []
    # iterate over classes
    for class_num in range(20):
        # no detection for the class? 0 ap.
        if len(detections[class_num]) == 0:
            aps.append(0.0); continue
        
        # sort detections and prefix sum
        confidence = sorted(detections[class_num])[::-1] #sort based on confidence
        corrects = np.array([positive for conf, positive in confidence])
        pfx = np.cumsum(corrects)
        precisions = pfx/(np.arange(len(pfx))+1)
        recalls = pfx/gt_cnts[class_num]
        
        # reverse-accumulate
        precisions_desc = np.maximum.accumulate(precisions[::-1])[::-1]
        
        # mAP@[0.05:0.95] COCO
        # reference
        # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
        ap = 0.0; cnt = 0;
        for recall_tick in np.arange(0.05, 1.05, 0.05):
            i = 0
            while i < len(recalls)-1 and recalls[i] < recall_tick: i += 1
            ap += precisions_desc[i] #recalls[i]~=recall_tick
            cnt += 1
        aps.append(ap/max(cnt, 1))
        
    mean_ap = 0
    for ap in aps: mean_ap += ap
    mean_ap /= max(1, len(aps))
    
    return mean_ap, aps

def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    :return ap:             mean average precision
            cls_aps[List]:  list of per-class average precision
    """
    # for each class store (pred_scorelist, T/F)
    detections = [[] for i in range(20)]
    # store how many instances of each class exists in valid set.(gt)
    gt_cnts = [0] * 20
    temp_precisions = [0] * 20
        
    print("evaluating on.. ", len(val_loader), " data")
    with torch.no_grad():
        for iter, data in tqdm.tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader)):
            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            rois = data['rois'].cuda()
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO (Q2.3): perform forward pass, compute cls_probs
            h, w = image.shape[-2:]
            scale_factor = rois.new([w, h, w, h])
            roi_pix = rois * scale_factor
            cls_probs = model.forward(image, roi_pix, target).detach().cpu().numpy()
            
            del image
            # TODO (Q2.3): Iterate over each class (follow comments)
            # ground truth detection counts.
            for cl in gt_class_list:
                gt_cnts[cl.item()] += 1
                
            # precompute bounding boxes.
            # each batch has only one instance, hence index [0]
            box_ious = box_iou(rois[0].cpu(), gt_boxes[0]).detach().numpy()
            
            # iterate over classes
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                cls_scores = cls_probs[:, class_num]
                used = [False] * len(gt_boxes)
                match_in_img = 0.0 # per-image precision, as a rough measure
                
                # iterate over detections
                for idx in range(len(rois)):
                    # filter low scores (negatives)
                    if cls_scores[idx] < thresh: continue
                    
                    # iterate over gt_bboxes and find matching bbox
                    gt_exists = False
                    for gt_idx in range(len(gt_boxes)):
                        if gt_class_list[gt_idx] != class_num or used[gt_idx]: continue
                        gt_exists = True
                        
                        if  box_ious[idx, gt_idx] > 0.5:
                            # a true positive
                            detections[class_num].append((cls_scores[idx], 1))
                            temp_precisions[class_num] += 1.0
                            used[gt_idx] = True
                        else:
                            # if not enough overlap, again it's false
                            detections[class_num].append((cls_scores[idx], 0))
                            
                    # if no gt_bbox for this class, it's a FP (i.e. image does not contain cls.)
                    if not gt_exists:
                        detections[class_num].append((cls_scores[idx], 0))      
                            
        print("---- eval: class-wise precision % ----")
        for c, cl in enumerate(CLASS_NAMES):
            print(f"{cl}: prec {temp_precisions[c]/max(1.0, gt_cnts[c]):.04f}({temp_precisions[c]}/{gt_cnts[c]})")
        ap, cls_aps = calculate_map(detections, gt_cnts)
        
    return ap,cls_aps
        

def train_model(model, train_loader=None, val_loader=None, optimizer=None, schedular=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    global USE_WANDB
    global CLASS_NAMES
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    batch_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(args.epochs):
        end = time.time()
        for iter, data in tqdm.tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):

            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            rois = data['rois'].cuda()
            
            h, w = image.shape[-2:]
            scale_factor = rois.new([w, h, w, h])
            rois = rois * scale_factor
            
            # take care that proposal values should be in pixels
            model.forward(image, rois, target)

            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedular.step()

            # average loss
            losses.update(loss.item())
            
            # measure average elapsed time per each forward
            batch_time.update(time.time() - end)
            end = time.time();
            
        # log epoch training info
        print(f'Epoch: [{epoch+1}]\tLR {optimizer.param_groups[0]["lr"]}\tLoss {losses.avg:.4f}')
        
        # validation per epoch
        model.eval()
        
        ap, cls_aps = test_model(model, val_loader)
        print("Epoch {0} Evaluation\t"
              "mAP {1:.04}".format(epoch+1, ap))

        wdb_imgs= []
        # use NMS to get boxes and scores for vis
        for n, data in enumerate(val_loader):
            # run forward for 10 images
            image = data['image'].cuda()
            target = data['label'].cuda()
            rois = data['rois'].cuda()

            h, w = image.shape[-2:]
            scale_factor = rois.new([w, h, w, h])
            roi_pix = rois * scale_factor
            cls_probs = model.forward(image, roi_pix, target).detach().cpu()
            
            # iterate over each class and run nms
            nms_bboxes = []
            for class_num in range(20):
                cls_scores = cls_probs[:, class_num]
                bboxes, scores = nms(rois[0].cpu(), cls_scores, 0.3)
                nms_bboxes.extend([{
                            "position": {
                                "minX": bbox[0],
                                "minY": bbox[1], #top(min row)
                                "maxX": bbox[2],
                                "maxY": bbox[3], #bottom(max row)
                                },
                            "class_id": class_num,
                            "box_caption": CLASS_NAMES[class_num],
                        } for bbox in bboxes])
                            
            # draw random 10 images with bounding boxes
            if USE_WANDB:
                viz_image = tensor_to_PIL(data['image'].squeeze(0))
                if len(nms_bboxes)>0:
                    wdb_imgs.append(wandb.Image(
                        viz_image,
                        boxes = { "proposals": {
                            "box_data": nms_bboxes
                        }}
                    ))
                else:
                    wdb_imgs.append(wandb.Image(viz_image))
            
            if n == 19: break # total 10 images shown. end of viz image loop
            
        model.train()

        if USE_WANDB:
            dic = {f"eval/ap_{CLASS_NAMES[cl]}":cls_ap for cl, cls_ap in enumerate(cls_aps)}
            dic.update({"eval/mAP": ap,  "train/loss": losses.avg, "train/lr":optimizer.param_groups[0]["lr"]})
            dic.update({"region proposals": wdb_imgs})
            dic.update({"epoch": epoch+1})
            
            wandb.log(dic)
        

def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    global USE_WANDB
    global CLASS_NAMES
    
    args = parser.parse_args()
    
    # config items to log
    config = {
        'lr': args.lr,
        'lr_decay_every':args.lr_decay_steps,
        'lr_decay': args.lr_decay,
        'momentum': args.momentum,
        'spp_sizes': [4, 2, 1],
        "run_id": "use spp"
    }
    
    # Initialize wandb logger
    if USE_WANDB:
        run = wandb.init(
            name = config["run_id"], 
            reinit = True, 
            # run_id = ### Insert specific run id here if you want to resume a previous run
            # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
            project = "weakly supervised deep detection network", 
            config=config
        )
        
    train_dataset = VOCDataset('trainval', image_size=512, top_n=300) #change to 300
    val_dataset = VOCDataset('test', image_size=512, top_n=300) #change to 300
    class_names = val_dataset.CLASS_NAMES
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(spp_dim=[4, 2, 1], use_spp=True,classes=train_dataset.CLASS_NAMES)
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
            print('Copied {}'.format(name))
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
            #print("froze ", name)
        elif 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
            #print("init ", name)
  
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay)

    # Training
    train_model(net, train_loader, val_loader, optimizer, scheduler, args)
    if USE_WANDB:
        run.finish()

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


