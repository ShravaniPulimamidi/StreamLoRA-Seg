import torch

from models.streamlora import StreamLoRA
from adaptation.updater import OnlineUpdater
from adaptation.entropy import entropy_loss

model = StreamLoRA()

updater = OnlineUpdater(model)

images = torch.randn(2, 3, 224, 224)

prediction = model(images)

loss = entropy_loss(prediction)

print("Loss before update:", loss.item())

updater.step(loss)

print("Model updated successfully.")