import torch

from models.streamlora import StreamLoRA

model = StreamLoRA()

x = torch.randn(2, 3, 224, 224)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)