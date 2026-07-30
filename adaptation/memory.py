import torch


class MemoryBank:
    """
    FIFO Memory Bank for continual adaptation.
    """

    def __init__(self, max_size=256):

        self.max_size = max_size
        self.memory = []

    def add(self, feature):

        feature = feature.detach().cpu()

        self.memory.append(feature)

        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_memory(self):

        if len(self.memory) == 0:
            return None

        return torch.stack(self.memory)

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)