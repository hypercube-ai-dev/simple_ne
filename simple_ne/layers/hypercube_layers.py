import torch.nn as nn
import torch

class FeedForward():
    def __init__(self, weights):
        self.ff = nn.Linear(weights.shape[0], weights.shape[1])
        # flatten and set weights