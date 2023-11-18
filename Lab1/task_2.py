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

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL
from PIL import Image, ImageDraw


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


def calculate_map():
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    pass


def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
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
            h, w = image.shape[-2:]
            scale_factor = torch.Tensor([w, h, w, h])
            rois = [
                torch.cat(bbox)*scale_factor for bbox in rois
            ]
            rois  = [torch.stack(rois, dim=0).float().cuda()]
            cls_probs = model.forward(image, rois, target)
            
            # TODO (Q2.3): Iterate over each class (follow comments)
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                
                # use NMS to get boxes and scores
                pass

            # TODO (Q2.3): visualize bounding box predictions when required
            calculate_map()


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    for epoch in range(args.epochs):
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
            rois  = [torch.stack(rois, dim=0).float().cuda()]
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

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", ap)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout

    # TODO (Q2.4): Plot class-wise APs


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    train_dataset = VOCDataset('trainval', image_size=512, top_n=10)
    val_dataset = VOCDataset('test', image_size=512, top_n=10)

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
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
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
            
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum)

    # Training
    train_model(net, train_loader, val_loader, optimizer, args)


if __name__ == '__main__':
    main()
