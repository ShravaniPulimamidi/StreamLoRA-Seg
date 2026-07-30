import torch

from models.clip_model import CLIPBackbone

model = CLIPBackbone()
model.eval()

dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    feat = model(dummy)

print("Feature shape:", feat.shape)