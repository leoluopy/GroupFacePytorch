import sys, os
import math
import torch
from pylab import plt

PI = 3.1415926


def get_target_lr(epoch, epoch_stride=20):
    return (math.cos(PI * (float(epoch % epoch_stride) / float(epoch_stride))) + 1) / 2


def adjust_learning_rate(optimizer, epoch, leak_lr, bottom_lr):
    set_lr = get_target_lr(epoch) * (leak_lr - bottom_lr) + bottom_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = set_lr
    return set_lr


if __name__ == '__main__':
    from torchvision.models import resnet18

    model = resnet18(pretrained=False)

    leak_lr = 1e-4
    bottom_lr = 1e-6

    optimizer = torch.optim.Adam(model.parameters(), lr=leak_lr)

    x = []
    y = []
    for epoch in range(200):
        adjust_learning_rate(optimizer, epoch, leak_lr, bottom_lr)
        for param_group in optimizer.param_groups:
            print("epoch: {} lr set to : {}".format(epoch, param_group['lr']))

        lr = get_target_lr(epoch) * leak_lr
        y.append(lr)
        x.append(epoch)
        plt.plot(x, y)
        plt.show()
