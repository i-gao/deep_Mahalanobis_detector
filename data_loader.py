# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# densenet transform
# TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
#     (125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)), ])
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])



def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(data_type, batch_size, data_path='./data'):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)

    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, data_path='./data'):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'imagenet_resize':
        data_path = os.path.expanduser('./data/Imagenet_resize')
        testsetout = datasets.ImageFolder(data_path, transform=TRANSFORM)
        test_loader = torch.utils.data.DataLoader(
            testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        data_path = os.path.expanduser('./data/LSUN_resize')
        testsetout = datasets.ImageFolder(data_path, transform=TRANSFORM)
        test_loader = torch.utils.data.DataLoader(
            testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader

