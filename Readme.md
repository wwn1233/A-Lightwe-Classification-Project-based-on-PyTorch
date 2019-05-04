# A project for classification problem,
# Weinong Wang, weinong.wang@hotmail.com

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
## Characteristics
 1. basic data augmentation: horizontal/vertical  flip, random rot (90), color jitter, random erasing, test augmentation
 2.  multi backbones: Resnet, Desnsenet et. al
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