from dataclasses import dataclass

@dataclass
class Config:
    # ==========================
    # Dataset
    # ==========================
    dataset = "BDD100K"

    train_root = "./data/BDD100K/train"
    val_root = "./data/BDD100K/val"

    image_size = (512, 512)
    num_classes = 19

    # ==========================
    # Model
    # ==========================
    backbone = "ViT-B-16"
    pretrained = True

    # ==========================
    # LoRA
    # ==========================
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1

    # ==========================
    # Training
    # ==========================
    batch_size = 4
    learning_rate = 1e-4
    weight_decay = 1e-4

    epochs = 20

    # ==========================
    # Test-Time Adaptation
    # ==========================
    entropy_weight = 1.0

    memory_size = 256

    adaptation_steps = 1

    # ==========================
    # Hardware
    # ==========================
    device = "cuda"

    num_workers = 4

    seed = 42


cfg = Config()