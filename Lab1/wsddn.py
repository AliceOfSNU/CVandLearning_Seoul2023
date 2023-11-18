import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        # feature extraction is same as AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.roi_pool = roi_pool
        
        
        self.classifier = self.classifier = nn.Sequential(
            # fc6
            nn.Linear(2 * 2 * 256, 2 * 256),
            nn.ReLU(inplace=True),
            # fc7
            nn.Linear(2 * 256, 256),
            nn.ReLU(inplace=True)
        )


        # fc8c
        self.score_fc = nn.Linear(256, self.n_classes)
        # fc8d
        self.bbox_fc = nn.Linear(256, self.n_classes)

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):


        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        x = self.features(image)
        x = self.roi_pool(x, rois, 2, spatial_scale=31/512) # N_roi * n_features *  2 * 2
        x = self.classifier(x.view(*x.shape[:-3], 256*2*2)) # N_roi * 256
        c_scores = self.score_fc(x).softmax(dim=-1) # softmax over classes
        d_scores = self.bbox_fc(x).softmax(dim=-2) # softmax over regions
        cls_prob = c_scores * d_scores # elementwise prod, N_roi * n_classes

        if self.training:
            label_vec = gt_vec.view(-1, self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        # labels mapped to -1, 1
        label_vec = (label_vec * 2) - 1.0
        # calculate loss, summing over regions
        summed_cls_probs = cls_prob.sum(0, keepdim=True)
        loss = torch.log(label_vec * (summed_cls_probs - 0.5) + 0.5)
        loss = loss.sum()

        return loss
