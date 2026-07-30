from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    def __init__(self, root, weather="fog", split="train", transform=None):

        self.root = Path(root)
        self.weather = weather
        self.split = split
        self.transform = transform

        self.image_dir = self.root / weather / split
        self.mask_dir = self.root / "gt" / weather / split

        self.images = sorted(self.image_dir.rglob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = Image.open(image_path).convert("RGB")

        # Placeholder mask for now
        # We will connect the real masks after verifying their filename pattern.
        mask = Image.new("L", image.size)

        if self.transform:
            image = self.transform(image)

        return image, mask