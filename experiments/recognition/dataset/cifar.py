##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong Wang
## Tencent, Youtu
## Email: weinong.wang@hotmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.utils.data as data
# import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

# from PIL import Image
import os
import os.path
import numpy as np
import math
import cv2
import random

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

# class RandomErasing(object):
#     """ Randomly selects a rectangle region in an image and erases its pixels.
#         'Random Erasing Data Augmentation' by Zhong et al.
#         See https://arxiv.org/pdf/1708.04896.pdf
#     Args:
#          probability: The probability that the Random Erasing operation will be performed.
#          sl: Minimum proportion of erased area against input image.
#          sh: Maximum proportion of erased area against input image.
#          r1: Minimum aspect ratio of erased area.
#          mean: Erasing value.
#     """

#     def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
#         self.probability = probability
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1

#     def __call__(self, img):
#         random_value = np.random.random()  # random.random()
#         # print(random_value)
#         # print(random_value)
#         if random_value > self.probability:
#             return img
#         # count = 0
#         for attempt in range(100):
#             area = img.size()[1] * img.size()[2]

#             target_area = np.random.uniform(self.sl, self.sh) * area
#             aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))

#             if w < img.size()[2] and h < img.size()[1]:
#                 # count +=1
#                 x1 = np.random.randint(0, img.size()[1] - h)
#                 y1 = np.random.randint(0, img.size()[2] - w)
#                 if img.size()[0] == 3:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
#                     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
#                 else:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                 return img
#         return img
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class Dataloder():
    def __init__(self, args):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        transform_train = transforms.Compose([
            # transforms.Resize(399),
            # transforms.RandomResizedCrop(224),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.2,0.2,0.2),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
            RandomErasing(),
        ])
        transform_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
        else:
            raise KeyError('dataset is not existing!')

        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            args.batch_size, shuffle=True, **kwargs)

        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            args.test_batch_size, shuffle=False, **kwargs)

       
        self.trainloader = trainloader 
        self.testloader = testloader
    
    def getloader(self):
        return self.trainloader, self.testloader

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

if __name__ == "__main__":
    trainset = MINCDataloder(root=os.path.expanduser('/media/youtu/本地磁盘1/data2/minc'), train=True)
    testset = MINCDataloder(root=os.path.expanduser('/media/youtu/本地磁盘1/data2/minc'), train=False)
    print(len(trainset))
    print(len(testset))
