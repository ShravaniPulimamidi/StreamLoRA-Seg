import torch
import torch.nn as nn


class SegmentationHead(nn.Module):

    def __init__(self, in_channels=512, num_classes=19):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)