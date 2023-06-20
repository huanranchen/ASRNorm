import torch
import torch.nn as nn
from math import cos, pi


class CosineLRS():
    def __init__(self, optimizer, max_epoch=300, lr_min=0, lr_max=0.05, warmup_epoch=30):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = lr_min
        self.max_lr = lr_max
        self.warmup_epoch = warmup_epoch

    def step(self, current_epoch):
        if current_epoch < self.warmup_epoch:
            lr = self.max_lr * current_epoch / self.warmup_epoch
        else:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + cos(pi * (current_epoch - self.warmup_epoch) /
                        (self.max_epoch - self.warmup_epoch))) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            print(f'now lr = {lr}')






