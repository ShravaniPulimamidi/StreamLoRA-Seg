import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """
    Simple semantic segmentation head.
    """

    def __init__(self,
                 in_channels=512,
                 num_classes=19):

        super().__init__()

        self.decoder = nn.Sequential(

            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):

        x = self.decoder(x)

        return x