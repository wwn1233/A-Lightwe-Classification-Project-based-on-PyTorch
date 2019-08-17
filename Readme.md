# A lightweight project for classification problem and bag of tricks are employed for better performance
### Weinong Wang, weinong.wang@hotmail.com

# News
2019.07.15 add OHEM and Mixup methods

2019.06.06 add warm up method of lr

2019.05.28 add reduced-resnet and different optimizers for cifar and fashionminist dtatset

2019.05.11 add support for cifar and fashionmnist dataset

# Requirements

Trained and Tested on Python3.5

	pytorch = 0.4.0
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

+ CIFAR100. In this experiment, we choose the **reduced-resne**t as our backbone network(you can choose yours).

||Models|Base|+RE|+Mixup
|---|---|---|---|---
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
 2. Multiple backbones
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
+ MINC-2500. The data structure is following the Materials in Context Database (MINC)
 -  data/minc-2500
     - images
     - labels
+ CIFAR100. The data would be automaticly downloaded to the folder：  "./data"

## Train
+ MINC-2500
python experiments/recognition/main.py - -dataset minc - -loss CrossEntropyLoss - -nclass  23 - -backbone resnet50 - -checkname test - -ocsm
+ CIFAR100
python experiments/recognition/main.py - -backbone resnet_reduce - -res_reduce_depth 20 - -solver_type SGD - -lr-step 200,300 - -dataset cifar100 - -lr 0.1 - -epochs 375 - -batch-size 384 - -mixup

Note: (- -lr-step 200,300) indicates that leanrning rate is decayed by 10 at 200-th and 300-th epoch; (- -lr-step 200,)  indicates that learning rate is decayed by 10 evary 200 epochs. (- - batch-size 384 - -ohem 192) indicates choosing 192 hard examples from 384 instances.


## Test
+ MINC-2500. For example:

python experiments/recognition/main.py - -dataset minc - -nclass  23 - -backbone resnet18 - -test-batch-size 128 - -eval  --resume  experiments/recognition/runs/minc/deepten/09-3/*.pth
+ CIFAR100. For example:

python experiments/recognition/main.py - -backbone resnet_reduce - -res_reduce_depth 20 - -dataset cifar100 - -test-batch-size 128 - -eval  --resume experiments/recognition/runs/cifar100/deepten/0/*.pth
 
## Related Repos
[PyTorch Encoding][PyTorch Encoding]

[Random Erasing][RE]


[PyTorch Encoding]:https://github.com/zhanghang1989/PyTorch-Encoding

[RE]:https://github.com/zhunzhong07/Random-Erasing
[Deep-TEN]:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Deep_TEN_Texture_CVPR_2017_paper.pdf
