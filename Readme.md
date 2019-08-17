# A lightweight project for classification problem and bag of tricks are employed for better performance
### Weinong Wang, weinong.wang@hotmail.com

# News
2019.07.15 add OHEM and Mixup methods

2019.06.06 add warm up method of lr

2019.05.28 add reduce-resnet and different optimizers for cifar and fashionminist dtatset

2019.05.11 add support for cifar and fashionmnist dataset

# Requirements

Trained and Tested on Python3.6
 1. pytorch = 0.4.0
	torchvision>=0.2.0
	matplotlib
	numpy
	scipy
	opencv
	pyyaml
	packaging
	PIL
	tqdm
	time

## Main Results
+ MINC-2500 is a patch classification dataset with 2500 samples per category. This is a subset of MINC where samples have been sized to 362 x 362 and each category is sampled evenly. Error rate and five fold cross validation are employed for evaluating. Based on resnet50, we can achieve a comparable result with state-od-the-arts.

||train1-vali1|train1-test1|train2-vali2|train2-test2|train3-vali3|train3-test3|train4-vali4|train4-test4|train5-vali5|train5-test5|Average
|---|---|---|---|---|---|---|---|---|---|---|---
|[Deep-TEN]|-|-|-|-|-|-|-|-|-|-|19.4%
|ours|19.0%|19.0%|19.0%|19.0%|19.0%|18.0%|19.0%|19.0%|20.0%|19.0%|19.0%

+ CIFAR100. In this experiment, we choose the reduced resnet as our backbone network(you can choose yours).

||Models|Base|+RE|+Mixup
|---|---|---|---|---|---|---|---|---|---|---|---
|[RE]|ResNet-20|30.84%|29.87%|-
|ours|ResNet-20|29.85%|28.61%|27.7%
+ More dataset coming soon ......

## Characteristics
 1. Basic data augmentation methods
	- horizontal/vertical flip
	- random rot (90)
	- color jitter
	- random erasing
	- test augmentation
	- lighting noise
	- mixup
 2.  Multiple backbones
 	- Resnet
	- Desnsenet
	- Reduced-resnet
 3. Other methods
	- Focal loss
	- Label smooth
	- Combining global max pooling and global average pooling
	- Orthgonal center loss based on subspace masking
	- Learning rate warmup
	- OHEM(online hard example mining)

			
## Data Preparation
The data structure is following the Materials in Context Database (MINC)
 -  minc-2500
     - images
     - labels
## Train
 CUDA_VISIBLE_DEVICES=0 python experiments/recognition/main.py --dataset minc --loss CrossEntropyLoss --nclass  2 --backbone resnet18 --checkname test --ocsm


## Test
 - Genaral version: 
 CUDA_VISIBLE_DEVICES=0 python experiments/recognition/main.py --dataset minc --loss CrossEntropyLoss --nclass  2 --backbone resnet18 --eval  --resume experiments/recognition/runs/minc/deepten/09-3/*.pth
 
## Related Repos
[PyTorch Encoding][PyTorch Encoding]

[Random Erasing][RE]


[PyTorch Encoding]:https://github.com/zhanghang1989/PyTorch-Encoding

[RE]:https://github.com/zhunzhong07/Random-Erasing
[Deep-TEN]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Deep_TEN_Texture_CVPR_2017_paper.pdf
