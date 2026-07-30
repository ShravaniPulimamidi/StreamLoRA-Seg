import torch
import torch.nn as nn

from models.clip_model import CLIPBackbone
from models.lora import LoRALayer
from models.segmentation_head import SegmentationHead


class StreamLoRASeg(nn.Module):
    """
    StreamLoRA-Seg:
    CLIP Backbone
          ↓
       LoRA Adapter
          ↓
    Segmentation Head
    """

    def __init__(self, num_classes=19):
        super().__init__()

        # CLIP image encoder
        self.backbone = CLIPBackbone()

        # LoRA adaptation
        self.lora = LoRALayer(
            in_features=512,
            out_features=512,
            rank=8
        )

        # Segmentation decoder
        self.segmentation_head = SegmentationHead(
            in_channels=512,
            num_classes=num_classes
        )

    def forward(self, x):

        # Extract CLIP features
        features = self.backbone(x)

        # Apply LoRA adaptation
        features = self.lora(features)

        # Segmentation prediction
        output = self.segmentation_head(features)

        return output