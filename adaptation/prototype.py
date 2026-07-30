import torch


class PrototypeBank:
    """
    Maintains a running prototype feature.
    """

    def __init__(self):
        self.prototype = None

    def update(self, features):
        """
        features: N x D
        """
        if features is None:
            return

        mean_feature = features.mean(dim=0)

        if self.prototype is None:
            self.prototype = mean_feature
        else:
            # Exponential moving average
            self.prototype = 0.9 * self.prototype + 0.1 * mean_feature

    def get(self):
        return self.prototype

    def reset(self):
        self.prototype = None