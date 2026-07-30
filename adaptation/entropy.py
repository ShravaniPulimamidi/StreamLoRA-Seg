import torch


def entropy_loss(prediction):
    """
    Entropy minimization for test-time adaptation.

    prediction:
        B x C x H x W
    """

    probability = torch.softmax(prediction, dim=1)

    entropy = -(probability * torch.log(probability + 1e-8)).sum(dim=1)

    return entropy.mean()