import torch
from torch import nn


class PhenotypeLayerFactory():

    def __init__(self):
        self.phenotype_dict = {
            "conv1d": self.to_conv1d,
            "conv2d": self.to_conv2d,
            "attention": self.to_attention
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
    
    def to_attention(weights):
        raise NotImplementedError