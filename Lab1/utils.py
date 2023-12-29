import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.ops as ops

# TODO: given bounding boxes and corresponding scores, perform non max suppression
# the tensors should be on cpu.
def nms(bounding_boxes, confidence_score, threshold=0.3):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    boxes, scores = None, None
    # ignore low confidence
    mask = confidence_score.gt(0.05)
    scores = confidence_score[mask]
    boxes = bounding_boxes[mask]
    if scores.shape[0] > 0:
        keep_idxs = ops.nms(boxes, scores, threshold) # 
        boxes = torch.index_select(boxes, 0, keep_idxs).detach().cpu().numpy().astype(np.float64)
        scores = torch.index_select(scores, 0, keep_idxs).detach().cpu().numpy().astype(np.float64)
    else: 
        boxes = []
        scores = []
    # USING torchvision.ops.nms instead
    return boxes, scores


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    # USING BOX_IOU in torchvision.ops
    iou = 1.0
    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id": classes[i],
        } for i in range(len(classes))
        ]

    return box_list


def VOC_collate_fn(batch):
    # dictionary collate function
    # rois, gt_boxes, gt_classes are converted to lists.
    keys = ['rois', 'label', 'wgt', 'image', 'gt_boxes', 'gt_classes']
    dic = {k:[] for k in keys}
    for bat in batch:
        for k in keys:
            dic[k].append(bat[k])
        
    dic['image'] = torch.stack(dic['image'])
    dic['wgt'] = torch.stack(dic['wgt'])
    dic['label']= torch.stack(dic['label'])

    return dic
    