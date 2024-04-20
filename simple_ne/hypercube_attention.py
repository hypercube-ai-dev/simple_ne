import copy
import numpy as np
import itertools
from math import factorial
import torch
#encodes a substrate of input and output coords with a cppn, adding 
#hidden coords along the 

class HyperAttention:

    def __init__(self, substrate, cppn, params, sgd_phenotypes=False):
        self.substrate = substrate
        self.cppn_q = cppn[0]
        self.cppn_k = cppn[1]
        self.cppn_v = cppn[2]
        self.cppn_o = cppn[3]
        self.cppn_ff = cppn[4]
        self.cppn_class = cppn[5]
        self.cppn_in_net = cppn[6]
        self.num_heads = 2 ** self.substrate.dim # number of children for a single subdivision
        self.head_depth = params["head_depth"] # how many times to subdivide at each head
        self.seq_len = params["sequence_len"]
        self.params = {}
        self.activation_string = params["activation"]
        self.max_weight = params["max_weight"]
        self.lvl = 0
        self.sgd_phenotypes = sgd_phenotypes

    def reset_substrate(self, substrate):
        self.substrate = substrate

    # creates phenotype transformer
    def create_phenotype_network_nd(self, coord_dictionary):
        input_coords = coord_dictionary["input_coords"]
        attn_coords = coord_dictionary["attn_coords"]
        ff_coords = coord_dictionary["ff_coords"]
        output_coords = coord_dictionary["output_coords"]
        mlp_coords = coord_dictionary['mlp_coords']
        attn_layers = [self.encode_attn_block(attn_coords[x], attn_coords[x], ff_coords[x]) for x in range(len(attn_coords))]
        classifier = self.endcode_output_net(attn_coords[-1], mlp_coords, output_coords)
        to_hidden_dim = query_torch_cppn_tensors(input_coords, attn_coords[0], True, self.cppn_in_net, self.max_weight).T
        transformer = TrasnformerClassifier(attn_layers, classifier, to_hidden_dim, self.seq_len)
        return transformer
        

    def endcode_output_net(self, in_coords, hidden_coords, out_coords):
        ff_weights = {}
        ff_weights["ff1"] = query_torch_cppn_tensors(in_coords, hidden_coords, True, self.cppn_class, self.max_weight).T
        ff_weights["ff2"] = query_torch_cppn_tensors(hidden_coords, out_coords, True, self.cppn_class, self.max_weight).T
        return FeedForward(ff_weights)


    def encode_attn_block(self, in_coords, attn_coords, ff_coords):
        attn_weights = {}
        ff_weights = {}
        qw = query_torch_cppn_tensors(in_coords, attn_coords, True, self.cppn_q, self.max_weight).T
        kw = query_torch_cppn_tensors(in_coords, attn_coords, True, self.cppn_k, self.max_weight).T
        vw = query_torch_cppn_tensors(in_coords, attn_coords, True, self.cppn_o, self.max_weight).T
        attn_weights["qkv_proj"] = torch.cat((qw, kw, vw), -1)
        attn_weights["o_proj"] = query_torch_cppn_tensors(attn_coords, attn_coords, True, self.cppn_o, self.max_weight).T
        ff_weights["ff1"] = query_torch_cppn_tensors(attn_coords, ff_coords, True, self.cppn_ff, self.max_weight).T
        ff_weights["ff2"] = query_torch_cppn_tensors(ff_coords, attn_coords, True, self.cppn_ff, self.max_weight).T
        return EncoderLayer(attn_weights, ff_weights, qw.shape[1], self.num_heads)

    def get_attention_layer(self, out_coords):
        inputs = self.substrate.input_coordinates
        self.encode_heads(inputs, out_coords)
        return

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
    inputs = get_nd_coord_inputs(coords_in, coords_out)
    activs = cppn(inputs)
    return activs

# if batchsize is left at -1 it will treat the coords as a single full batch
def get_nd_coord_inputs(in_coords, out_coords, batch_size=-1):
    in_expanded = []
    out_expanded = []

    #it is expected the last dimension will be the coordinate space (1 for 1d coords, 2 for 2d etc)
    for i in range(in_coords.shape[-1])
