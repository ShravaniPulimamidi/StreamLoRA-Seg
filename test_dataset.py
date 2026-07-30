from datasets.bdd100k import ACDCDataset
from datasets.transforms import get_train_transform

dataset = ACDCDataset(
    root="data",
    weather="fog",
    split="train",
    transform=get_train_transform()
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask size:", mask.size)