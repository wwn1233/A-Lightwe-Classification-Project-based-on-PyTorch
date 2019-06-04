##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong Wang
## Tencent, Youtu
## Email: weinong.wang@hotmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Deep Encoding')

        parser.add_argument('--dataset', type=str, default='minc',
            help='training dataset (default: cifar10)')
        parser.add_argument('--basepath', type=str, default='./experiments/recognition/',
                            help='group dataset name(default: coat_length_labels)')
        parser.add_argument('--loss', type=str,
                            default='CrossEntropyLoss',
                            help='please choose loss function')

        parser.add_argument('--nclass', type=int, default=23, metavar='N',
                            help='number of classes (default: 10)')
        parser.add_argument('--backbone', type=str, default='resnet50, resnet_reduce',
                            help='backbone name (default: inceptionresnetv2)')
        parser.add_argument('--res_reduce_depth', type=int, default=20,
                            help='num of blocj in resnet_reduce for cifar (20, 32, 44, 56 , 110)')
        parser.add_argument('--batch-size', type=int, default=32,
                            metavar='N', help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=128,
                            metavar='N', help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=60, metavar='N',
                            help='number of epochs to train (default: 300)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--plot', action='store_true', default=False,
                            help='matplotlib')
        parser.add_argument('--checkname', type=str, default='0',
                            help='set the checkpoint name')
        # model params 
        parser.add_argument('--model', type=str, default='deepten',
            help='network model type (default: densenet)')

        # training hyper params
        parser.add_argument('--start_epoch', type=int, default=0,
            metavar='N', help='the epoch number to start (default: 0)')
        # lr setting
        parser.add_argument('--solver_type', type=str, default='SGD', 
            help='solver type (SGD, Adam, Rmsprop)')
        parser.add_argument('--lr-scheduler', type=str, default='step', 
            help='learning rate scheduler (default: step)')
        parser.add_argument('--lr-step', type=str, default='30,', metavar='LR',
            help='learning rate step (default: 40)')
        # optimizer
        parser.add_argument('--momentum', type=float, default=0.9, 
            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, 
            metavar ='M', help='SGD weight decay (default: 5e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', 
            default=False, help='disables CUDA training')

        parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
            help='put the path to resuming file if needed')

        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
            help='evaluating')

        parser.add_argument('--test_aug', action='store_true', default=False,
                            help='matplotlib')
        parser.add_argument('--ocsm', action='store_true', default=False,
                            help='orthogonal center learning for subspace learning')

        # warm up
        parser.add_argument('--warmup', action='store_true', default=False,
                            help='')
        parser.add_argument('--warmup_epoch', type=int, default=18,
                            , help='epochs for warm up')
        parser.add_argument('--warmup_factor', type=float, default=0.01,
                            , help='the start lr for warm up')
        


        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
