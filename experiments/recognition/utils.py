##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import shutil
import os
import sys
import time
import math
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

##Added by Weinong Wang, Tencent. ( CrossEntropyLabelSmooth, FocalLoss, CenterLoss_5)
__all__ = ['get_optimizer', 'LR_Scheduler', 'save_checkpoint', 'progress_bar', 'CrossEntropyLabelSmooth','FocalLoss', 'CenterLoss_5']

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True, num_classes = None, epsilon = 0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # if isinstance(alpha, Variable):
        self.alpha = Variable(alpha * torch.ones(num_classes, 1))
        # else:
        #     self.alpha = Variable(alpha)
        self.size_average = size_average
        self.class_num = num_classes
        

    def forward(self, inputs, target):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CenterLoss_5(nn.Module):
    """

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True): #, margin = 0.3):
        super(CenterLoss_5, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu


        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        # self.margin = margin


    def forward(self, x, labels, center_weight):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        
        # print(loss_center_inter)
        #
        loss_all = 0

        return loss_all



def get_optimizer(args, model, diff_LR=True):
    """
    Returns an optimizer for given model, 

    Args:
        args: :attr:`args.lr`, :attr:`args.momentum`, :attr:`args.weight_decay`
        model: if using different lr, define `model.pretrained` and `model.head`.
    """
    if diff_LR and model.pretrained is not None:
        print('Using different learning rate for pre-trained features')
        if args.solver_type == 'SGD':
            optimizer = torch.optim.SGD([
                        {'params': model.pretrained.parameters()}, 
                        {'params': model.head.parameters(), 
                          'lr': args.lr*10},
                    ], 
                    lr=args.lr,
                    momentum=args.momentum, 
                    weight_decay=args.weight_decay)
        elif args.solver_type == 'Adam':
            optimizer = torch.optim.Adam([
                        {'params': model.pretrained.parameters()}, 
                        {'params': model.head.parameters(), 
                          'lr': args.lr*10},
                    ], 
                    lr=args.lr,
                    weight_decay=args.weight_decay)
        elif args.solver_type == 'Rmsprop':
            optimizer = torch.optim.RMSprop([
                        {'params': model.pretrained.parameters()}, 
                        {'params': model.head.parameters(), 
                          'lr': args.lr*10},
                    ], 
                    lr=args.lr,
                    momentum=args.momentum, 
                    weight_decay=args.weight_decay)
        else:
            raise KeyError("Unsupported optim: {}".format(args.solver_type))
    else:
        if args.solver_type == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=args.momentum, 
                                        weight_decay=args.weight_decay) 
            # print(model.parameters())
        elif args.solver_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),  weight_decay= args.weight_decay)
        elif args.solver_type == 'Rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay) 
        else:
            raise KeyError("Unsupported optim: {}".format(args.solver_type))
    return optimizer

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`), :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs, :attr:`args.lr_step`

        niters: number of iterations per epoch
    """
    def __init__(self, args, niters=420):
        self.mode = args.lr_scheduler 
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = args.lr
        if self.mode == 'step':
            self.lr_step = eval(args.lr_step)
        else:
            self.niters = niters
            self.N = args.epochs * niters
        self.epoch = -1

        if self.mode == 'step' and len(self.lr_step) > 1:
            self.decay_steps_ind = 0
            for i in range(0, len(self.lr_step)):
                # print(self.lr_step)
                # print()
                if self.lr_step[i] >= args.start_epoch:
                    self.decay_steps_ind = i
                    break

    def __call__(self, optimizer, i, epoch, best_pred):
        if self.mode == 'cos':
            T = (epoch - 1) * self.niters + i
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            # print(lr)
        elif self.mode == 'poly':
            T = (epoch - 1) * self.niters + i
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            if len(self.lr_step) == 1:
                lr = self.lr * (0.1 ** ((epoch - 1) // self.lr_step[0]))
            else:
                # print(epoch == self.lr_step[self.decay_steps_ind])
                if self.decay_steps_ind < len(self.lr_step) and \
                    epoch == self.lr_step[self.decay_steps_ind]:
                    # print('WWN')
                    lr = self.lr * 0.1
                    self.lr = lr # update self.lr
                    self.decay_steps_ind +=1
                else:
                    lr = self.lr
        else:
            raise RuntimeError('Unknown LR scheduler!')
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (
                epoch, lr, best_pred))
            self.epoch = epoch
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1,len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "%s/runs/%s/%s/%s/"%(args.basepath, args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


# refer to https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
# terms_height, term_width = os.popen('stty size', 'r').read().split()
term_width=2
term_width = int(term_width)-1
TOTAL_BAR_LENGTH = 36.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    """Progress Bar for display
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()    # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('    Step: %s' % _format_time(step_time))
    L.append(' | Tot: %s' % _format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def _format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
