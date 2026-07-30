from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import get_train_transform


class ACDCDataset(Dataset):
    def __init__(self, root="data", weather="fog", split="train"):
        self.root = Path(root)
        self.weather = weather
        self.split = split

        self.image_dir = self.root / weather / split
        self.label_dir = self.root / "gt" / weather / split

        self.transform = get_train_transform()

        self.images = sorted(
            self.image_dir.rglob("*_rgb_anon.png")
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        label_name = image_path.name.replace(
            "_rgb_anon.png",
            "_gt_labelTrainIds.png"
        )

        label_path = image_path.parent.parent.parent

        folder = image_path.parent.name

        label_path = (
            self.label_dir /
            folder /
            label_name
        )

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image = self.transform(image)

        return image, label