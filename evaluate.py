import torch

from torch.utils.data import DataLoader

from datasets.acdc import ACDCDataset
from models.streamlora import StreamLoRA
from configs.config import cfg


def evaluate():

    dataset = ACDCDataset(
        root=cfg.data_root,
        split="val"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = StreamLoRA(
        num_classes=cfg.num_classes
    ).to(cfg.device)

    model.eval()

    total = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(cfg.device)

            outputs = model(images)

            print(outputs.shape)

            total += 1

    print("Validation Images:", total)


if __name__ == "__main__":
    evaluate()