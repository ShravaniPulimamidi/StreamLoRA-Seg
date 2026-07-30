import torch

from models.segmentation_head import SegmentationHead

model = SegmentationHead()

x = torch.randn(2, 512, 32, 32)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)