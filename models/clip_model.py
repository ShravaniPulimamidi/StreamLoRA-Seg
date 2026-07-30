import torch
import open_clip


class CLIPBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k"
        )

    def forward(self, image):
        return self.model.encode_image(image)