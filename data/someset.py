'''
this file aims to read any dataset satisfied that:
    1.all the images are in one folder
    2.only a dict to store ground truth. Keys are image names, values are ground truth labels.
'''

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class SomeDataSet(Dataset):
    def __init__(self, img_path, gt_path):
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1)),
            # transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        self.images = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
        self.gt = np.load(gt_path, allow_pickle=True).item()
        self.img_path = img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        now = self.images[item]
        now_img = Image.open(os.path.join(self.img_path, now))  # numpy
        return self.transform(now_img), self.gt[now]


def get_someset_loader(img_path,
                       gt_path,
                       batch_size=128,
                       num_workers=8,
                       pin_memory=False, ):
    set = SomeDataSet(img_path=img_path, gt_path=gt_path)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory)
    return loader
