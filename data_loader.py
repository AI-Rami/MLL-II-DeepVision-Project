"""
data_loader.py
---------------
Handles all dataset preparation steps for CIFAR-10.

Responsibilities:
- Download CIFAR-10 automatically if not already present.
- Apply transformations:
    * Convert images to tensors.
    * Normalize RGB channels to range [-1, 1].
- Create DataLoader objects for training and testing sets.

Returned objects:
trainloader, testloader → used by the main training scripts.
"""


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    """Download CIFAR-10, normalize it, and return train/test DataLoaders."""

    # Define transformations (convert to tensor + normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 dataset: 60k color images (32x32, 10 classes)
    trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
