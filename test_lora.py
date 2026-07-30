import torch

from models.lora import LoRALayer

lora = LoRALayer(
    in_features=512,
    out_features=512,
    rank=8,
)

x = torch.randn(4, 512)

y = lora(x)

print("Input :", x.shape)
print("Output:", y.shape)