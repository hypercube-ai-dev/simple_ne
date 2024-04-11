import torch
from torch import nn


class PhenotypeLayerFactory():

    def __init__(self):
        self.phenotype_dict = {
            "conv1d": self.to_conv1d,
            "conv2d": self.to_conv2d  
        }

    def to_conv1d(weights):
        shape = weights.shape
        m = nn.Conv1d(shape[0], shape[1], shape[2], stride=1)
        m.weight = weights
        return m
    
    def to_conv2d(weights):
        shape = weights.shape
        m = nn.Conv2d(shape[0], shape[1], shape[2], stride=1)
        m.weight = weights
        return m

class ConvGru(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv_block = nn.Conv1d(in_size, 64, 4)
        self.drop_relu = nn.Sequential(
             nn.Dropout(.5), nn.Linear( 64, 64), nn.ReLU(inplace=True)
		)
        self.gru = torch.nn.GRU(64, 128, 3, batch_first=True)
        self.linear_out = nn.Linear(128, out_size)
        self.out_size = out_size

    def forward(self, input, h0=None):
        if h0 == None:
            h0 = torch.zeros(3, input.shape[0], 128).cuda()
        conv_out = self.conv_block(input)
        conv_out = conv_out.permute(0,2,1)
        gru_out, hn = self.gru(conv_out, h0)
        out = self.linear_out(gru_out)
        return out, hn