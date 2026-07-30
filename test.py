import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.acdc import ACDCDataset
from models.streamlora import StreamLoRA


def main():

    train_dataset = ACDCDataset(
        root=cfg.data_root,
        split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    model = StreamLoRA(
        num_classes=cfg.num_classes
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss()

    print("Training started...")

    for epoch in range(cfg.epochs):

        model.train()

        total_loss = 0

        for images, masks in train_loader:

            images = images.to(cfg.device)
            masks = masks.to(cfg.device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{cfg.epochs} "
            f"Loss: {total_loss:.4f}"
        )


if __name__ == "__main__":
    main()