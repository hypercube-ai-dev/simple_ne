import torch
from torch import nn

class LinearNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
             nn.Linear(in_size, hidden_size),
             nn.ReLU(inplace=True),
             nn.Linear( hidden_size, out_size),
             nn.ReLU(inplace=True)
		)

    def forward(self, input):
        return self.net(input)