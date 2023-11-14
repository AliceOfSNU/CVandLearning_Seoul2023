import torch.nn as nn
import torchvision.models as models
import torch

import sys
sys.path.append(".")

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'alexnet/local': 'CVandLearning_Seoul2023/Lab1/model/alexnet_pretrained_wts.pth'
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        # TODO (Q1.1): Define model
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
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model


    def forward(self, x):
        # TODO (Q1.7): Define forward pass

        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not
    if pretrained:
        state_dic = torch.load(model_urls['alexnet/local'])
        new_state_dic = {}
        for k, v in state_dic.items():
            if 'features' in k:
                new_state_dic[k.partition('features.')[2]] = v
        model.features.load_state_dict(new_state_dic)
    else:
        for k,m in model.featurs.named_parameters():
            if 'weight' in k:
                nn.init.xavier_uniform_(m)
    # initialize classifier with xavier.
    for k, m in model.classifier.named_parameters():
        if 'weight' in k:
            nn.init.xavier_uniform_(m)
        
    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not

    return model
