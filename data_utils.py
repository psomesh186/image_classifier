import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_dataloader(root_dir):
    """Creates dataloader for required task.
    Args:
        root_dir (str): Directory with all images
    
    Returns:
        A dataloader object.
    """
    transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
