import torch
from torch import nn
import layers.general_layers as gl

class FutureFM(torch.Module):
    def __init__(self, 
                 in_size: int,
                 patch_size: int,
                 out_size: int,
                 embed_size: int, 
                 dropout: float = 0.0, 
                 max_len: int = 5000):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = gl.ResBlock(in_size, embed_size, embed_size, dropout)
        self.p_encoder = gl.PositionalEncoding(embed_size)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, 8),
            8
        )
        self.out_layer = gl.ResBlock(embed_size, out_size, out_size, dropout)

    def forward(self, input):
        x = self.in_layer(input)
        x = self.p_encoder(x)
        x = self.decoder(x, is_casual=True)
        return self.out_layer(x)