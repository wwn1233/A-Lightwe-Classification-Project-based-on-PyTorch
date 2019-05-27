##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong Wang
## Tencent, Youtu
## Email: weinong.wang@hotmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as resnet
from .resnet_reduce import resnet_reduce

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)

class Net(nn.Module):
    def __init__(self, args):
        nclass=args.nclass
        super(Net, self).__init__()
        self.backbone = args.backbone
        self.dataset = args.dataset
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True)
            self.feat_num = 2048
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True)
            self.feat_num = 2048
        elif self.backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True)
            self.feat_num = 2048
        elif self.backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=True)
            self.feat_num = 512
        elif self.backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=True)
            self.feat_num = 1024
        elif self.backbone == 'resnet_reduce':
            self.depth = args.res_reduce_depth
            self.pretrained = resnet_reduce(depth = self.depth)
            self.feat_num = self.pretrained.block.expansion * 64
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        if self.dataset == 'cifar10' or self.dataset == 'cefar100':
            self.classifier = nn.Linear(self.feat_num,nclass)
            # self.classifier.apply(weights_init_classifier)

        else:
            self.linear = nn.Linear(self.feat_num, 512)
            self.head = nn.Sequential(
                # nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )

            self.head.apply(weights_init_kaiming)

            self.classifier = nn.Linear(512,nclass)
            self.classifier.apply(weights_init_classifier)

        self.softmax = nn.Softmax(1).cuda()

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x 
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))
        # print(x)
        if self.backbone == 'resnet_reduce':
            x = self.pretrained.res1(x)  
            x = self.pretrained.res2(x)
            x = self.pretrained.res3(x)
            x = self.pretrained.res4(x)
        else:
            x = self.pretrained.conv1(x)  
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            # print(x)
            # print(torch.sum(torch.abs(x -0.355846) < 0.00001))
            x = self.pretrained.maxpool(x)
            x = self.pretrained.layer1(x)
            x = self.pretrained.layer2(x)
            x = self.pretrained.layer3(x)
            x = self.pretrained.layer4(x)

        if self.dataset == 'cifar10' or self.dataset == 'cefar100': # simple net for cifar
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(x.size(0), -1)
            f = None
            x= self.classifier(x)
        else:
           x1 = F.max_pool2d(x, x.size()[2:])
           x2 = F.avg_pool2d(x, x.size()[2:])
           x = x1 + x2
           x = x.view(x.size(0), -1)
           f = self.linear(x)
           x = self.head(f)
           # print(x.size())
           x= self.classifier(x)

        return x, f, self.classifier.weight#self.softmax(x)
