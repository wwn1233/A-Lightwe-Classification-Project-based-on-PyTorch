# A lightweight project for classification problem and bag of tricks are employed for better performance
### Weinong Wang, weinong.wang@hotmail.com

# News
2019.06.06 add warm up method of lr

2019.05.28 add reduce-resnet and different optimizers for cifar and fashionminist dtatset

2019.05.11 add support for cifar and fashionmnist dataset

# Requirements

Trained and Tested on Python3.6
 1. pytorch >= 0.4.0
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
MINC-2500 is a patch classification dataset with 2500 samples per category. This is a subset of MINC where samples have been sized to 362 x 362 and each category is sampled evenly. Error rate is employed for evaluating. We can achieve a comparable results only based on resnet50 with state-od-the-arts.

|backbone|train1-vali1|train2-vali2|train3-vali3|train4-vali4|train5-vali5
|---|---|---|---|---|---
|resnet18|19%|19%|19%|19%|20%

|backbone|train1-test1|train2-test2|train3-test3|train4-test4|train5-test5
|---|---|---|---|---|---
|resnet18|19%|19%|18%|19%|19%
## Characteristics
 1. basic data augmentation: horizontal/vertical  flip, random rot (90), color jitter, random erasing, test augmentation
 2.  multi backbones: Resnet, Desnsenet, reduce-resnet et. al
 3. Focal loss; Label smooth; combining global max pooling and global average pooling; our orthgonal center loss based on subspace masking

			
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
