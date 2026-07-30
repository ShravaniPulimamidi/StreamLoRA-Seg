import torch
import open_clip


class CLIPBackbone(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k"
        )

    def encode_image(self, image):

        return self.model.encode_image(image)

    def encode_text(self, text):

        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        tokens = tokenizer(text)

        return self.model.encode_text(tokens)

    def forward(self, image):

        return self.encode_image(image)