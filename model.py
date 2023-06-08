import torch.nn as nn


class CNNClassifier(nn.Module):
    """An image classifier using CNN."""

    def __init__(self, num_labels):
        """
        Args:
            num_labels (int): Number of classes present in dataset.    
        """
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5
                ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(in_features=256, out_features=num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x