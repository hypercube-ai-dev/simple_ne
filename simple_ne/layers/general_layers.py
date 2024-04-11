import torch
from torch import nn
import math

class ResBlock(nn.Module):
    def __init__(self, 
                 input_size, 
                 out_size, 
                 hidden_size, 
                 drop=0.0):
        super(ResBlock, self).__init__()
        self.lin_res = nn.Linear(input_size, out_size)
        self.hidden = nn.Sequential(
            nn.Linear(input_size, out_size),
            nn.ReLU(inplace=True)
        )
        if drop != 0.0:
            self.drop = nn.Dropout(drop)

    def forward(self, input):
        rx = self.lin_res(input)
        x = self.hidden(input)
        return x + rx
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
