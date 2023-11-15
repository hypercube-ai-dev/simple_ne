import torch
import itertools

class SubDivisionCube(object):
    def __init__(self, dim, initial_depth, center, width):
        self.width = width
        self.center = center
        self.dim = dim
        self.initial_depth = initial_depth
        self.signs = list(itertools.product([1,-1], repeat=len(dim)))
        self.children = {}
        self.initial_divide()

    def initial_divide(self):
        depth = 0
        while depth < self.initial_depth:
            depth += 1
            num_children = (2**self.dim) ** depth
            offsets = torch.ones(num_children, self.dim) * (self.width / (2*depth))
        return