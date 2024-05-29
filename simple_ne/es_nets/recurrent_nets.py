import torch
from torch import nn

class GruNet(torch.Module):
    def __init__(self, in_size, out_size, gru_layers=3, gru_in=64, gru_out=128):
        super().__init__()
        self.lin_in = nn.Linear(in_size, gru_in)
        self.gru = torch.nn.GRU(gru_in, gru_out, gru_layers, batch_first=True)
        self.lin_out = nn.Linear(128, out_size)
        self.out_size = out_size
        self.gru_layers = gru_layers
        self.in_size = in_size
        self.gru_out = gru_out
        self.gru_in = gru_in

    def forward(self, x: torch.Tensor, h0=None):
        if h0 == None:
            h0 = torch.zeros(self.gru_layers, self.in_size, self.gru_out).to(x.get_device())
        out = self.lin_in(x)
        out, h0 = self.gru(out, h0)
        out = self.lin_out(out)
        return out, h0
