import torch

from adaptation.entropy import entropy_loss

prediction = torch.randn(2, 19, 32, 32)

loss = entropy_loss(prediction)

print("Entropy Loss:", loss.item())