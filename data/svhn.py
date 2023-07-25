from torchvision.datasets import SVHN
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ['get_svhn_test', 'get_svhn_train']


def get_svhn_train(batch_size=256,
                   num_workers=40,
                   pin_memory=True,
                   ):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    set = SVHN('../resources/svhn/', split='train', download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=True)
    return loader


def get_svhn_test(batch_size=256,
                  num_workers=40,
                  pin_memory=True,
                  ):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    set = SVHN('../resources/svhn/', split='test', download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return loader
