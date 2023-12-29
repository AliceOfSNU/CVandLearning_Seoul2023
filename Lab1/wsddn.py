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
        
        # pooling and spp
        self.roi_pool = roi_pool
        self.roi_size = 32
        ## if pyramid_pool
        self.spp_sizes = [16, 4, 1]
        self.spp_dim = 0
        for sz in self.spp_sizes: self.spp_dim += sz
        ## else
        self.spp_dim_sqrt = 4
        self.spp_dim = 16
        ##
        self.classifier = nn.Sequential(
            # fc6
            nn.Linear(self.spp_dim * 256, self.spp_dim * 64),
            nn.ReLU(inplace=True),
            # fc7
            nn.Linear(self.spp_dim * 64, 256),
            nn.ReLU(inplace=True)
        )


        # fc8c (classification head)
        self.score_fc = nn.Linear(256, self.n_classes)
        # fc8d (detection head)
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
        ## if spp_pooling
            # spp-pooling is a pyramid like max pooling
            # each pyramid layer outputs different dims, 4*4, 2*2, 1*1, for example
            # they are concatenated and flattened to a 21*feature_dim - tensor
        ## else
        x = self.roi_pool(x, [rois], self.spp_dim_sqrt, spatial_scale=31/512) # N_roi * n_features *  2 * 2
        ## endif
        
        x = self.classifier(x.view(*x.shape[:-3], 256*self.spp_dim)) # N_roi * 256
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
        #label_vec = (label_vec * 2) - 1.0
        # calculate loss, summing over regions
        summed_cls_probs = cls_prob.sum(0, keepdim=True)
        summed_cls_probs = summed_cls_probs.clamp(0.0, 1.0)
        #loss = torch.log(label_vec * (summed_cls_probs - 0.5) + 0.5)
        #loss = loss.sum()
        loss = F.binary_cross_entropy(summed_cls_probs, label_vec, reduction='sum')
        return loss
