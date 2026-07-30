import torch

from adaptation.memory import MemoryBank

memory = MemoryBank(memory_size=5)

for i in range(7):
    feature = torch.randn(512)
    memory.add(feature)
    print(f"Added feature {i+1}")

print("\nMemory size:", len(memory))

stored = memory.get_memory()

print("Stored tensor shape:", stored.shape)