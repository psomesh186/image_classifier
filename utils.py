import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(root_dir, shuffle=True, batch_size=64, num_workers=4):
    """Creates dataloader for required task.

    Args:
        root_dir (str): Directory with all images.
        shuffle (bool): Toggle dataloader shuffle. (Default: True)
        batch_size (int): Batch size of the mini-batches. (Default: 64)
        num_workers (int): Number of parallel workers. (Default: 4)

    Returns:
        A dataloader object.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return dataloader, dataset.class_to_idx

def compute_accuracy(outputs, labels):
    """Computes accuracy of prediction.
    
    Args:
        outputs (Tensor): The output tensors given by model.
        labels (Tensor): Ground truth labels.
    """
    outputs = torch.argmax(outputs, dim=1)
    acc = (outputs == labels).sum().item() / len(outputs)
    return acc