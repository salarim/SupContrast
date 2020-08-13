from __future__ import print_function

import os
import sys
import argparse
import time
import math
import copy

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, MLP
from byol_loss import BYOLLoss
from main_supcon import set_loader

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'shapenet'], help='dataset')
    parser.add_argument('--data-folder', type=str, default='./datasets/')
    parser.add_argument('--views', type=int, default=2,
                        help='views')
    parser.add_argument('--drop-objects-ratio', type=float, default=0.0,
                        help='Drop objects ratio')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_v_{}_dr_{}'.\
        format('BYOL', opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial,
               opt.views, opt.drop_objects_ratio)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_models(opt):
    online_encoder = SupConResNet(name=opt.model)
    online_predictor = MLP()
    criterion = BYOLLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        online_encoder = apex.parallel.convert_syncbn_model(online_encoder)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            online_encoder.encoder = torch.nn.DataParallel(online_encoder.encoder)
        online_encoder = online_encoder.cuda()
        online_predictor = online_predictor.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    target_encoder = copy.deepcopy(online_encoder)

    models = {'online_encoder': online_encoder,
              'target_encoder': target_encoder,
              'online_predictor': online_predictor}

    return models, criterion


def set_optimizer(opt, models):
    optimizer = optim.SGD(list(models['online_encoder'].parameters()) + \
                          list(models['online_predictor'].parameters()),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def update_moving_average(ma_model, current_model, beta=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = old_weight * beta + (1 - beta) * up_weight


def train(train_loader, models, criterion, optimizer, epoch, opt):
    """one epoch training"""
    for model in models.values():
        model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = images[0].shape[0]
        views = len(images)
        images = torch.cat(images, dim=0)
        images = images.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        online_projs = models['online_encoder'](images)
        online_preds = models['online_predictor'](online_projs)
        with torch.no_grad():
            target_projs = models['target_encoder'](images).detach()


        online_preds = torch.split(online_preds, [bsz]*views, dim=0)
        online_preds = [f.unsqueeze(1) for f in online_preds]
        online_preds = torch.cat(online_preds, dim=1)

        target_projs = torch.split(target_projs, [bsz]*views, dim=0)
        target_projs = [f.unsqueeze(1) for f in target_projs]
        target_projs = torch.cat(target_projs, dim=1)

        loss = criterion(online_preds, target_projs)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target model
        update_moving_average(models['target_encoder'], models['online_encoder'])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build models and criterion
    models, criterion = set_models(opt)

    # build optimizer
    optimizer = set_optimizer(opt, models)

    # tensorboard
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, models, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(models['online_encoder'], optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(models['online_encoder'], optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
