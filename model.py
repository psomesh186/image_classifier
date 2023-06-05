import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """An image classifier using CNN."""

    def __init__(self, num_labels):
        """
        Args:
            num_labels (int): Number of classes present in dataset.    
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=7,
                stride=2,
                bias=False
                ),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                bias=False,
                stride=2
                ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                bias=False,
                ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=512),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.out(x)
        return x