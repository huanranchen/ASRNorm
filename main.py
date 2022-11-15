from tester import test_acc
import torch

from data import get_CIFAR10_test

loader = get_CIFAR10_test()
from torchvision.models import resnet50
from backbones import resnet32
from Normalizations import ASRNormBN, ASRNormIN

model = resnet32(num_classes=10, norm_layer=ASRNormBN)
model.load_state_dict(torch.load('./student.pth'))
test_acc(model, loader)
