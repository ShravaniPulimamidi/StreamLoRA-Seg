import torch

from models.streamlora import StreamLoRASeg


model = StreamLoRASeg()

dummy = torch.randn(2, 3, 224, 224)

output = model(dummy)

print("Input :", dummy.shape)
print("Output:", output.shape)