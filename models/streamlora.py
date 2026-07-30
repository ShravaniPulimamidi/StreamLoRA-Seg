import torch
import torch.nn as nn

from models.clip_model import CLIPBackbone
from models.lora import LoRALayer
from models.segmentation_head import SegmentationHead


class StreamLoRA(nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()

        self.backbone = CLIPBackbone()

        self.lora = LoRALayer(
            in_features=512,
            out_features=512
        )

        self.segmentation_head = SegmentationHead(
            in_channels=512,
            num_classes=num_classes
        )

    def forward(self, images):

        # CLIP image encoder
        features = self.backbone(images)

        # Apply LoRA
        features = self.lora(features)

        # Convert feature vector to feature map
        B = features.shape[0]

        features = features.view(B, 512, 1, 1)

        features = torch.nn.functional.interpolate(
            features,
            size=(32, 32),
            mode="bilinear",
            align_corners=False
        )

        # Pixel-wise prediction
        output = self.segmentation_head(features)

        return output