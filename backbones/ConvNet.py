import torch
import torch.nn as nn

__all__ = ['convnet']


class ConvNet(nn.Module):
    def __init__(self, dim=1, norm_layer=nn.BatchNorm2d, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(dim, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = norm_layer(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


def convnet(dim, norm_layer, num_classes):
    return ConvNet(dim, norm_layer, num_classes)


