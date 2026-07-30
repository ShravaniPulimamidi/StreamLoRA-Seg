import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Basic Low-Rank Adaptation (LoRA) module.
    """

    def __init__(self, in_features, out_features,
                 rank=8,
                 alpha=16,
                 dropout=0.1):

        super().__init__()

        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(dropout)

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

        self.scaling = alpha / rank

    def forward(self, x):

        return self.B(self.dropout(self.A(x))) * self.scaling