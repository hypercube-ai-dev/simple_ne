import copy
import numpy as np
import itertools
from math import factorial
import torch
from .layers.hypercube_layers import TrasnformerClassifier, FeedForward, EncoderLayer
from simple_ne.sub_cube import SubDivisionCube
#encodes a substrate of input and output coords with a cppn, adding 
#hidden coords along the 

class HyperAttention:

    def __init__(
            self, 
            substrate, 
            params,
            initial_depth,
            center,
            width, 
            sgd_phenotypes=False):
        self.substrate = substrate
        self.num_heads = 2 ** len(center) # number of children for a single subdivision
        self.head_depth = params["head_depth"] # how many times to subdivide at each head
        self.seq_len = params["sequence_len"]
        self.params = {}
        self.max_weight = params["max_weight"]
        self.lvl = 0
        self.sgd_phenotypes = sgd_phenotypes
        self.cube = SubDivisionCube(center, initial_depth, width)
    
    def encode_input_layer(self, in_coords, net):
        x = get_nd_coords_new(in_coords, self.cube.tree[0])
        return net(x)

    # use this to encode weights between any two depths of the subdivision tree
    # these can be the same depth if desired
    def encode_hiddens(self, from_depth, to_depth, net):
        x = get_nd_coords_new(self.cube.tree[from_depth], self.cube.tree[to_depth])
        return net(x)

    def encode_output_layer(self, from_depth, out_coords, net):
        x = get_nd_coords_new(self.cube.tree[from_depth], out_coords)
        return net(x)

    def reset_substrate(self, substrate):
        self.substrate = substrate
        
    def endcode_feedforward_net(self, in_coords, hidden_coords, out_coords, cppn):
        ff_weights = {}
        ff_weights["ff1"] = query_torch_cppn_tensors(in_coords, hidden_coords, True, cppn, self.max_weight).T
        ff_weights["ff2"] = query_torch_cppn_tensors(hidden_coords, out_coords, True, cppn, self.max_weight).T
        return FeedForward(ff_weights)

# a tree that subdivides n dimensional euclidean spaces
class BatchednDimensionTree:
    
    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.child_coords = []
        self.cs = []
        self.signs = self.set_signs() 
        self.child_weights = 0.0
        self.divided = False
    def set_signs(self):
        return list(itertools.product([1,-1], repeat=len(self.coord)))
    
    def divide_childrens(self):
        for x in range(self.num_children):
            new_coord = []
            for y in range(len(self.coord)):
                new_coord.append(self.coord[y] + (self.width/(2*self.signs[x][y])))
            self.child_coords.append(new_coord)
            newby = BatchednDimensionTree(new_coord, self.width/2, self.lvl+1)
            self.cs.append(newby)
        self.divided = True

    def get_coords_at_depth(self, depth, coords):
        if self.lvl == depth:
            if self.divided == False:
                self.divide_childrens()
            coords.extend(self.child_coords)
            #print(len(self.child_coords))
            return coords
        else:
            if self.divided == False:
                self.divide_childrens()
            for t in self.cs:
                coords = t.get_coords_at_depth(depth, coords)
            return coords
# new tree's corresponding connection structure
class nd_Connection:
    def __init__(self, coord1, coord2, weight):
        if(type(coord1) == list):
            coord1 = tuple(coord1)
        if(type(coord2) == list):
            coord2 = tuple(coord2)
        self.coord1 = coord1
        self.coord2 = coord2
        self.coords = coord1 + coord2
        self.weight = weight
    def __eq__(self, other):
        return self.coords == other.coords
    def __hash__(self):
        return hash(self.coords + (self.weight,))

def query_torch_cppn_tensors(coords_in, coords_out, outgoing, cppn, max_weight=5.0):
    inputs = get_nd_coord_inputs_as_tensor(coords_in, coords_out)
    activs = cppn(inputs)
    return activs

def get_nd_coords_new(in_coords, out_coords):
    in_expanded = in_coords.unsqueeze(1).expand(-1, out_coords.shape[0], -1).reshape(-1, in_coords.shape[-1])
    out_expanded = out_coords.repeat(in_coords.shape[0],1)
    return torch.cat((in_expanded, out_expanded), dim=1)

def get_nd_coord_inputs_as_tensor(in_coords : torch.tensor, out_coords : torch.tensor):
    in_expanded = in_coords.unsqueeze(1)
    out_expanded = out_coords.unsqueeze(0)
    combined = torch.cat((in_expanded.expand(-1, out_coords.shape[0], -1), out_expanded.expand(in_coords.shape[0], -1, -1)), dim=2)
    return combined.view(-1, in_coords.shape[-1] * 2)

def get_nd_coord_inputs_as_dict(in_coords, out_coords, batch_size=None):
    n_in = len(in_coords)
    n_out = len(out_coords)
    num_dimens = len(in_coords[1])
    dimen_arrays = {}

    if batch_size is not None:
        in_coords = in_coords.unsqueeze(0).expand(batch_size, n_in, num_dimens)
        out_coords = out_coords.unsqueeze(0).expand(batch_size, n_out, num_dimens)

        for x in range(num_dimens):
            dimen_arrays[str(x) + "_out"] = out_coords[:, :, x].unsqueeze(2).expand(batch_size, n_out, n_in) 
            dimen_arrays[str(x) + "_in"] = in_coords[:, :, x].unsqueeze(1).expand(batch_size, n_out, n_in)
    else:
        for x in range(num_dimens):
            dimen_arrays[str(x) + "_out"] = out_coords[:, x].unsqueeze(1).expand(n_out, n_in) 
            dimen_arrays[str(x) + "_in"] = in_coords[:, x].unsqueeze(0).expand(n_out, n_in)
    return dimen_arrays