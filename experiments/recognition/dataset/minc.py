##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong Wang
## Tencent, Youtu
## Email: weinong.wang@hotmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.utils.data as data
# import torchvision
from torchvision import transforms

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

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        random_value = np.random.random()  # random.random()
        # print(random_value)
        if random_value > self.probability:
            return img
        # count = 0
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                # count +=1
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

class RandomHorizontalFlip(object):
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img[:,::-1,:].copy()
        return img

class RandomVerticalFlip(object):
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img[::-1,:,:].copy()
        return img

class RandomRot90(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.transpose(img, [1,0,2]).copy()
        return img
'''
hsv transformer：
hue_delta: the change ratio of hue
sat_delta: the change ratio of saturation
val_delta: the change ratio of value
'''
def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)

'''
random hsv transformer
hue_vari: the range of change ratio of hue
sat_vari: the range of change ratio of saturation
val_vari: the range of change ratio of value
'''
def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(filename, datadir, class_to_idx):
    images = []
    labels = []
    with open(os.path.join(filename), "r") as lines:
        for line in lines:
            _image = os.path.join(datadir, line.rstrip('\n'))
            # print(_image)
            _dirname = os.path.split(os.path.dirname(_image))[1]
            print(_image)
            assert os.path.isfile(_image)
            label = class_to_idx[_dirname]
            images.append(_image)
            labels.append(label)

    return images, labels

class MINCDataloder(data.Dataset):
    def __init__(self, root, train=True, transform=None, test_aug = False):
        self.transform = transform
        classes, class_to_idx = find_classes(root + '/images')
        if train:
            filename = os.path.join(root, 'labels/train1.txt')
        else:
            filename = os.path.join(root, 'labels/validate1.txt')

        self.images, self.labels = make_dataset(filename, root, 
            class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.train = train
        self.test_aug = test_aug

    def __getitem__(self, index):
        # _img = Image.open(self.images[index]).convert('RGB')
        _img = cv2.imread(self.images[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        if self.train:
            height,width,_ = _img.shape
            min_size = min(height,width)
            # max_size = max(height,width)
            scale = 256 / float(min_size)
            target_width = round(width * scale)
            target_height = round(height * scale)

            _img = cv2.resize(_img,(target_width, target_height), cv2.INTER_LINEAR)
            _img = RandomHorizontalFlip()(_img)
            # _img = RandomHorizontalFlip()(_img)
            _img = RandomVerticalFlip()(_img)
            _img = RandomRot90()(_img)
            _img = random_hsv_transform(_img, 0.4,0.4,0.4)
        else:
            height,width,_ = _img.shape
           # ratio_h = height / 720  # this setting is only for our real scenes, you can change it for your scenes
           # ratio_w = width / 1280  # this setting is only for our real scenes, you can change it for your scenes

           # target_width = int(ratio_w * 320)
           # target_height = int(ratio_h * 240)

           # ## crop from original images
           # _img = _img[-target_height:, int(math.floor(width / 2 - target_width/2)):int(math.floor(width / 2 + target_width/2))]
           # height,width,_ = _img.shape
            # print( _img.shape)
            # cv2.imwrite('test.jpg',_img)

            min_size = min(height,width)
            # max_size = max(height,width)
            scale = 256 / float(min_size)  # the minimum side is set 192, you can change it for your scenes
            target_width = round(width * scale)
            target_height = round(height * scale)

            _img = cv2.resize(_img,(target_width, target_height), cv2.INTER_LINEAR)
            if self.test_aug:
                    _images = [_img]
                    _images.append(np.fliplr(_img).copy())
                    _images.append(np.flipud(_img).copy())
                    _images.append(np.fliplr(_images[-1]).copy())
                    _images.append(np.transpose(_img, (1,0,2)).copy())
                    _images.append(np.flipud(_images[-1]).copy())
                    _images.append(np.fliplr(_images[-2]).copy())
                    _images.append(np.flipud(_images[-1]).copy())

        _label = self.labels[index]
        if self.transform is not None:
            if  self.test_aug:
                _img_t = []
                for i in range(len(_images)):
                    _img_t.append(self.transform(_images[i]))
            else:
                _img_t = self.transform(_img)
            if self.train:
                _img_t = RandomErasing(mean=[0.0, 0.0, 0.0])(_img_t)   #

        return  _img_t, _label #,self.images[index]

    def __len__(self):
        return len(self.images)

class Dataloder():
    def __init__(self, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            # transforms.Resize(399),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.2,0.2,0.2),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        transform_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = MINCDataloder(root=os.path.expanduser('./data/minc-2500'),
            train=True, transform=transform_train)
        testset = MINCDataloder(root=os.path.expanduser('./data/minc-2500'),
            train=False, transform=transform_test, test_aug = args.test_aug)
    
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            args.batch_size, shuffle=True, **kwargs)
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
