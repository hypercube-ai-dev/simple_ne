import torch
import torch.nn as nn
from .base import EsBase


class GruNet(EsBase):
    def __init__(self, in_size, out_size, gru_layers=2, gru_in=8, gru_out=16):
        super().__init__()
        self.lin_in = nn.Linear(in_size, gru_in)
        self.gru = torch.nn.GRU(gru_in, gru_out, gru_layers, batch_first=True)
        self.lin_out = nn.Linear(gru_out, out_size)
        self.out_size = out_size
        self.gru_layers = gru_layers
        self.in_size = in_size
        self.gru_out = gru_out
        self.gru_in = gru_in

    def forward(self, x: torch.Tensor, h0=None):
        if h0 == None:
            h0 = torch.zeros(self.gru_layers, self.gru_out).to(x.device)
        out = self.lin_in(x)
        out, h0 = self.gru(out, h0)
        out = self.lin_out(out)
        return out, h0