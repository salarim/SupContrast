from __future__ import print_function

import argparse
import numpy as np

import torch
from torchvision import transforms, datasets

from util import TwoCropTransform
from data_utils import ImageFolder


def parse_option():
    parser = argparse.ArgumentParser('argument for Computing dataset statistics')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'shapenet'], help='dataset')
    parser.add_argument('--data-folder', type=str, default='./datasets/')

    opt = parser.parse_args()

    return opt


def set_loader(opt, train=True):

    if opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.data_folder,
                                         train=train,
                                         transform=TwoCropTransform(transforms.ToTensor()),
                                         download=True)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root=opt.data_folder,
                                          train=train,
                                          transform=TwoCropTransform(transforms.ToTensor()),
                                          download=True)
    elif opt.dataset == 'shapenet':
        dataset = ImageFolder(root=opt.data_folder, transform=transforms.ToTensor())
    else:
        raise ValueError(opt.dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return data_loader


def main():
    opt = parse_option()

    data_loader = set_loader(opt)

    sum_x, sum_x2 = torch.zeros(3), torch.zeros(3)
    count = 0
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):

            images = torch.cat(images, dim=0)
            count += images.shape[0]
            sum_x += torch.sum(torch.mean(images, dim=(2, 3)), dim=0)
            sum_x2 += torch.sum(torch.mean(torch.pow(images, 2), dim=(2, 3)), dim=0)

            if idx % 10 == 0:
                print('{}/{}'.format(idx, len(data_loader)))
    
    mean = sum_x / count
    std = torch.sqrt(sum_x2 / count - mean**2)

    print(mean.tolist(), std.tolist())


if __name__ == '__main__':
    main()