import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
from torchvision import transforms
from tllib.vision.datasets import PACS
from tllib.vision.transforms import ResizeImage
from tllib.vision.datasets.imagelist import MultipleDomainsDataset

"""
install tllib:
git clone git@github.com:thuml/Transfer-Learning-Library.git
python setup.py install
pip install -r requirements.txt
"""


class NPACS(PACS):
    def __init__(self, root: str, task: str, split='all', download=True, **kwargs):
        super(NPACS, self).__init__(root, task, split, download, **kwargs)

    def __getitem__(self, index):
        img, target = super(NPACS, self).__getitem__(index)
        return img, target


def get_pacs_dataset(target_domain, root="./data/pacs", download=True, augment=True):
    assert target_domain in ["P", "A", "C", "S"]

    if augment:
        test_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((227, 227), scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.RandomGrayscale(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ]
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((227, 227), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ]
    )
    test_dataset = NPACS(root=root,
                         task=target_domain,
                         transform=test_transform,
                         download=download)

    source_domain = [i for i in ["P", "A", "C", "S"] if target_domain != i]

    train_dataset = []
    for domain in source_domain:
        train_dataset.append(NPACS(root=root,
                                   task=domain,
                                   transform=train_transform if augment else test_transform,
                                   download=download))
    train_dataset = ConcatDataset(train_dataset)
    return train_dataset, test_dataset


def get_PACS_train(batch_size=128,
                   num_workers=0,
                   pin_memory=True,
                   augment=True,
                   target_domain="A"
                   ):
    set, _ = get_pacs_dataset(root='./resources/PACS', target_domain=target_domain, augment=augment)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=True)
    return loader


def get_PACS_test(batch_size=128,
                  num_workers=0,
                  pin_memory=True,
                  augment=False,
                  target_domain="A"
                  ):
    _, set = get_pacs_dataset(root='./resources/PACS', target_domain=target_domain, augment=augment)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=True)
    return loader
