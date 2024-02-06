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

    def __init__(self, spp_dim, use_spp = True, classes=None):
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
        self.spp_size = 0
        self.use_spp = use_spp
        if use_spp:
            self.spp_dim = spp_dim
            for sz in spp_dim: self.spp_size += sz*sz
        else:
            self.spp_dim = spp_dim
            self.spp_size = spp_dim*spp_dim
        ##
        self.classifier = nn.Sequential(
            # fc6
            nn.Linear(self.spp_size * 256, self.spp_size * 64),
            nn.ReLU(inplace=True),
            # fc7
            nn.Linear(self.spp_size * 64, 256),
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

        # start with feature map of 256 channels and 31*31 size
        x = self.features(image) # 1(batch), channels=256, 31, 31
        if self.use_spp:
            # spp-pooling is a pyramid like max pooling
            # each pyramid layer outputs different dims, 4*4, 2*2, 1*1, for example
            # they are concatenated and flattened to a 21*feature_dim - tensor
            spp_outputs = []
            spp_outputs.append(self.roi_pool(x, [*rois], self.spp_dim[0], spatial_scale=31/512).flatten(-2))
            spp_outputs.append(self.roi_pool(x, [*rois], self.spp_dim[1], spatial_scale=31/512).flatten(-2))
            spp_outputs.append(self.roi_pool(x, [*rois], self.spp_dim[2], spatial_scale=31/512).flatten(-2))
            x = torch.cat(spp_outputs, -1).flatten(-2)
        else:            
            x = self.roi_pool(x, [*rois], self.spp_dim, spatial_scale=31/512) # N_roi * n_features *  2 * 2
            x = x.view(*x.shape[:-3], 256*self.spp_size)
        ## endif
        
        x = self.classifier(x) # N_roi as batch_size, flattened 1st dim
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
        # calculate loss, summing over regions
        summed_cls_probs = cls_prob.sum(0, keepdim=True)
        summed_cls_probs = torch.clamp(summed_cls_probs[0].unsqueeze(-1), 0.0, 1.0)
        label_vec = label_vec[0].unsqueeze(-1)
        loss = F.binary_cross_entropy(summed_cls_probs, label_vec, reduction='sum')
        return loss
