import torch


class PrototypeMemory:
    """
    Stores class prototypes computed from feature vectors.
    """

    def __init__(self):
        self.prototypes = {}

    def update(self, class_id, features):
        """
        Compute prototype as mean feature vector.
        """
        prototype = features.mean(dim=0)
        self.prototypes[class_id] = prototype.detach().cpu()

    def get(self, class_id):
        """
        Retrieve prototype for a class.
        """
        return self.prototypes.get(class_id, None)

    def clear(self):
        self.prototypes = {}

    def __len__(self):
        return len(self.prototypes)