import torch

from adaptation.memory import MemoryBank

memory = MemoryBank(max_size=5)

for i in range(7):
    feature = torch.randn(512)
    memory.add(feature)

print("Memory Size:", len(memory))

stored = memory.get_memory()

print("Stored Shape:", stored.shape)