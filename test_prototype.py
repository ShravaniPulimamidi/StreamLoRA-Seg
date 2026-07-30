import torch

from adaptation.prototype import PrototypeBank

prototype = PrototypeBank()

features = torch.randn(8, 512)

prototype.update(features)

print("Prototype Shape:", prototype.get().shape)

features2 = torch.randn(8, 512)

prototype.update(features2)

print("Updated Prototype Shape:", prototype.get().shape)