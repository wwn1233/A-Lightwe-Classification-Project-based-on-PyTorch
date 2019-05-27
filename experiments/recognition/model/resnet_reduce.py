from __future__ import absolute_import

'''reduced Resnet.
modidied by Weinong Wang, Youtu
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''

import torch.nn as nn
import math

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride =1):
    "3 x 3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride= stride, padding = 1, bias =False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x ):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias  = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernal_size = 3, stride = stride, \
                               padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernal_size = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_reduce(nn.Module):

    def __init__(self, depth, num_classes = 1000):
        super(ResNet_reduce, self).__init__()

        assert (depth -2) % 6 == 0, 'depth should be 6n+2'
        n = (depth -2) // 6

        self.block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.res1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 3, padding =1, bias = False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(16),)

        self.res2 = self._make_layer(self.block, 16, n)
        self.res3 = self._make_layer(self.block, 32, n, stride = 2)
        self.res4 = self._make_layer(self.block, 64, n, stride = 2)
        self.gap  = nn.AvgPool2d(8)
        self.fc = nn.Linear(64*self.block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, num_block, stride =1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride =stride, bias= False),
                nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x 

def resnet_reduce(**kwargs):
    return ResNet_reduce(**kwargs)