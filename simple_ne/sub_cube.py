import torch
import itertools

class SubDivisionCube(object):
    def __init__(self, dim, initial_depth, center):
        self.center = center
        self.dim = dim
        self.initial_depth = initial_depth
        self.signs = list(itertools.product([1,-1], repeat=len(dim)))
        self.initial_divide()

    def initial_divide(self):

        return