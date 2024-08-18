import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from .activations import str_to_activation


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class EncoderLayer(nn.Module):
    def __init__(self, transformer_weights, ff_weights, hidden_dim, num_heads=8):
        super().__init__()
        self.attention = AttentionLayer(transformer_weights, hidden_dim, num_heads)
        self.ff = FeedForward(ff_weights)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, mask=None):
        #with torch.no_grad():
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)
        linear_out = self.ff(x)
        x = x + linear_out
        x = self.norm2(x)
        return x

class TrasnformerClassifier(nn.Module):
    def __init__(self, encoder_layers, classifier, in_net, seq_len, with_grad=True):
        super().__init__()
        self.in_net = torch.nn.Parameter(in_net)
        self.layers = torch.nn.ModuleList(encoder_layers)
        self.classifier = classifier
        #self.pos_encoder = PositionalEncoding(in_net.size(-1), seq_len)

        if with_grad == False:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x, mask=None):
        x = torch.matmul(x, self.in_net)
        #with torch.no_grad():
        for l in self.layers:
            x = l(x, mask=mask)
        x = self.classifier(x)
        x = F.softmax(x, -1) 
        return x

class FeedForward(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.l1 = torch.nn.Parameter(weights['ff1'])
        self.l2 = torch.nn.Parameter(weights['ff2'])

    def forward(self, x):
        #with torch.no_grad():
        x = F.relu(torch.matmul(x, self.l1))
        return torch.matmul(x, self.l2)

class AttentionLayer(nn.Module):

    def __init__(self, weights, embed_dim, num_heads=8):
        super().__init__()
        self.qkv_proj = torch.nn.Parameter(weights["qkv_proj"])
        self.o_proj = torch.nn.Parameter(weights["o_proj"])
        self.num_heads = 8
        self.head_dim = embed_dim // self.num_heads
        self.embed_dim = embed_dim

    
    def forward(self, x, mask=None, return_attention=False):
        #with torch.no_grad():
        batch_size, seq_length, _ = x.size()
        qkv = torch.matmul(x, self.qkv_proj)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = torch.matmul(values, self.o_proj)

        if return_attention:
            return o, attention
        else:
            return o


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x