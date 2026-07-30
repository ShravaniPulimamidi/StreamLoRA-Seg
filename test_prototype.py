import torch

from adaptation.prototype import PrototypeMemory

prototype_memory = PrototypeMemory()

# Simulate features for class 0
features = torch.randn(10, 512)

prototype_memory.update(0, features)

prototype = prototype_memory.get(0)

print("Prototype shape:", prototype.shape)
print("Stored classes:", len(prototype_memory))
