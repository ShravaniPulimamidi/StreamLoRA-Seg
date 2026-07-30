import torch


class OnlineUpdater:

    def __init__(self, model, lr=1e-4):

        self.model = model

        # Update only trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(
            params,
            lr=lr
        )

    def step(self, loss):

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()