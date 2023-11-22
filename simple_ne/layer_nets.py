import torch
from torch import nn
from random import random
from .activations import activations


class SimpleNeLayer(nn.Module):
    def __init__(self, input_size, output_size, weights=None):
        super().__init__()
        if weights != None:
            self.weights = weights
        else:
            self.weights = torch.randn(output_size, input_size)

    def forward(self, inputs):
        return torch.matmul(inputs, self.weights)