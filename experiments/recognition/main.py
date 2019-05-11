##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong Wang
## Tencent, Youtu
## Email: weinong.wang@hotmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import os
import matplotlib.pyplot as plot
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from option import Options
from utils import *
from utils import CrossEntropyLabelSmooth
from tqdm import tqdm
from PIL import Image
import time
import numpy as np

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []

def main():
    # init the args
    global best_pred, errlist_train, errlist_val
    args = Options().parse()
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    # plot
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ## set the class number according to the dataset
    if args.dataset == 'minc':
        args.nclass = 23
        dataset_load = args.dataset
    elif args.dataset == 'cifar10':
        args.nclass = 10
        dataset_load = args.dataset[:-2]
    elif args.dataset == 'cifar100':
        args.nclass = 100
        dataset_load = args.dataset[:-3]
    elif args.dataset == 'fashionmnist':
        args.nclass = 10
        dataset_load = args.dataset
    else:
        dataset_load = args.dataset
        print('Remember to set the --nclass according to your dataset!')
    # init dataloader
    dataset = importlib.import_module('dataset.' + dataset_load)
    Dataloder = dataset.Dataloder
    train_loader, test_loader = Dataloder(args).getloader()


    # init the model
    models = importlib.import_module('model.' + args.model)
    model = models.Net(args)
    # criterion and optimizer
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'LabelSmooth':
        criterion = CrossEntropyLabelSmooth(num_classes=args.nclass)
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss(gamma=2, alpha=0.2, \
                                      num_classes= args.nclass,
                                      epsilon=0.1)

    if args.ocsm:
        criterion_center = CenterLoss_5(num_classes=args.nclass, feat_dim=512
                                    )
    else:
        criterion_center = None
    optimizer = get_optimizer(args, model, False)
    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = torch.nn.DataParallel(model)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            errlist_train = checkpoint['errlist_train']
            errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            for name, param in checkpoint['state_dict'].items():
                print(name)
        else:
            print("=> no resume checkpoint found at '{}'". \
                  format(args.resume))
    scheduler = LR_Scheduler(args, len(train_loader))

    def train(epoch):
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0, 0, 0
        # adjust_learning_rate(optimizer, args, epoch, best_pred)
        # print(type(train_loader))
        tbar = tqdm(train_loader, desc='\r')
        # train_loss_end=0
        batch_idx_end = 0
        # err_end=0
        # total_end=0
        # correct_end=0
        for batch_idx, (data, target) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            # print(batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output, f, center_weight = model(data)
            loss = criterion(output, target)
            if args.ocsm:
                center_loss = 0.01 * criterion_center(f, target, center_weight)
            else:
                center_loss = 0
            loss = loss + center_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)
            err = 100 - 100. * correct / total
            tbar.set_description('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
                                 (train_loss / (batch_idx + 1), err, total - correct, total))
            train_loss_end = train_loss
            batch_idx_end = batch_idx
            # err_end=err
            # total_end= total
            # correct_end=correct
        print('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
              (train_loss / (batch_idx_end + 1), err, total - correct, total))
        errlist_train += [err]

    def test(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0, 0, 0
        is_best = False
        tbar = tqdm(test_loader, desc='\r')

        batch_idx_end = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tbar):
                # print(target)
                # print(ids)
                if args.test_aug:
                    output = torch.zeros((1, 2)).cuda()
                    # print(data.size())
                    if args.cuda:
                        target = target.cuda()
                    target = Variable(target)
                    data_all = torch.cat([data[0], \
                                          data[1], \
                                          data[2], \
                                          data[3], \
                                          data[4], \
                                          data[5], \
                                          data[6], \
                                          data[7], \
                                          ], 0)
                    if args.cuda:
                        data_all = data_all.cuda()
                    data_all = Variable(data_all, volatile=True)
                    output_all, _, _ = model(data_all)
                    # print(output_all)
                    output += output_all.sum(0)  # +output_all[3]
                else:
                    # print(data.size())
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data, volatile=True), Variable(target)
                    output, _, _ = model(data)

                test_loss += criterion(output, target).data
                pred = output.data.max(1)[1]

                is_right = pred.eq(target.data).cpu().sum()

                correct += is_right
                total += target.size(0)

                err = 100 - 100. * correct / total
                tbar.set_description('Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                                     (test_loss / (batch_idx + 1), err, total - correct, total))
                batch_idx_end = batch_idx

        print('Loss: %.3f | Err: %.3f%%, (%d/%d)' % \
            (test_loss / (batch_idx_end + 1), err, total - correct, total))

        if args.eval:
            print('Error rate is %.3f' % err)
            return
        # save checkpoint
        errlist_val += [err]
        if err < best_pred:
            best_pred = err
            is_best = True
        print('Best Error rate is %.3f' % best_pred)

        time.sleep(10)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'errlist_train': errlist_train,
            'errlist_val': errlist_val,
        }, args=args, is_best=is_best)
        if args.plot:
            plot.clf()
            plot.xlabel('Epoches: ')
            plot.ylabel('Error Rate: %')
            plot.plot(errlist_train, label='train')
            plot.plot(errlist_val, label='val')
            plot.legend(loc='upper left')
            plot.draw()
            plot.pause(0.001)

    if args.eval:
        test(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(epoch)
        test(epoch)

    # save train_val curve to a file
    if args.plot:
        plot.clf()
        plot.xlabel('Epoches: ')
        plot.ylabel('Error Rate: %')
        plot.plot(errlist_train, label='train')
        plot.plot(errlist_val, label='val')
        plot.savefig("%s/runs/%s/%s/%s/" % (args.basepath, args.dataset, args.model, args.checkname)
                     + 'train_val.jpg')


if __name__ == "__main__":
    main()
