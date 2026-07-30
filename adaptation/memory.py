import torch


class MemoryBank:
    """
    FIFO memory bank for continual test-time adaptation.
    Stores feature vectors extracted from previous frames.
    """

    def __init__(self, memory_size=256):
        self.memory_size = memory_size
        self.features = []

    def add(self, feature):
        """
        Add a new feature vector.
        """
        feature = feature.detach().cpu()

        self.features.append(feature)

        if len(self.features) > self.memory_size:
            self.features.pop(0)

    def get_memory(self):
        """
        Return all stored features.
        """
        if len(self.features) == 0:
            return None

        return torch.stack(self.features)

    def clear(self):
        """
        Empty the memory.
        """
        self.features = []

    def __len__(self):
        return len(self.features)