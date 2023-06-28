import torch
from torch import nn
from .ALRS import ALRS


# cifar10-c WRN with 40 layers and widen factor 4
# SGD with Nestrov momentum 0.9 and batch_size 128
# learning 0.1 with cosine annealing 200 epoch
# def default_optimizer(model: nn.Module, lr=0.1) -> torch.optim.Optimizer:
#     return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-5)

def default_optimizer(model: nn.Module, lr=1e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


# ResNet-18 pretrained on ImageNet
# SGD with initial learning rate as 0.004 decays 10% after 24 epochs
# batch_size 128  30 peochs
def PACS_optimizer(model: nn.Module, lr=0.1, decay=0) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay, nesterov=True)


def default_lr_scheduler(optimizer):
    return ALRS(optimizer)
