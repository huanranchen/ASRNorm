from tester import test_acc
import torch

from data import get_CIFAR100_test

loader = get_CIFAR100_test()
from torchvision.models import resnet50
from Normalizations import ASRNormBN, ASRNormIN

model = resnet50(num_classes=100)
model.load_state_dict(torch.load('./student.pth'))
test_acc(model, loader)
