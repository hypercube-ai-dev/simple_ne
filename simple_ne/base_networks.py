import torch
from torch import nn
from random import random
from .activations import activations

class SimpleNENode(object):
    def __init__(self, activation, in_idxs, key, weights=None, is_output=False):
        self.activation = activation
        if weights == None:
            self.weights = torch.randn(len(in_idxs))
        else:
            self.weights = weights
        self.in_idxs = in_idxs
        self.node_key = key
        self.is_output = is_output

    def activate(self, inputs, batched=False):
        if batched == True:
            inputs = torch.index_select(inputs, 1, self.in_idxs)
        else:
            inputs = torch.index_select(inputs, 0, self.in_idxs)
        return self.activation(torch.matmul(self.weights, inputs))

class SimpleNEAgent(nn.Module):
    def __init__(self, nodes: list[SimpleNENode], input_size, output_size, batch_size = 0):
        super().__init__()
        self.nodes = nodes
        self.in_size = input_size
        self.out_size = output_size
        self.batch_size = batch_size
        self.reset()
        return

    def reset(self):
        # this is to track the value at each node
        if self.batch_size == 0:
            self.activs = torch.zeros(self.in_size + len(self.nodes)+1)
        else:
            self.activs = torch.zeros(self.batch_size, self.in_size + len(self.nodes)+1)
    # nodes are added in order and have a
    # probability of connecting to existing nodes
    # each node will need the output of nodes that come before it
    def forward(self, x, mask=None):
        if self.batch_size == 0:
            self.activs[:x.shape[0]] = x
            out = []
            for ix,n in enumerate(self.nodes):
                n_out = n.activate(self.activs)
                self.activs[x.shape[0]+ix] = n_out
                if n.is_output == True:
                    out.append(n_out)
        else:
            self.activs[:,:x.shape[1]] = x
            out = []
            for ix,n in enumerate(self.nodes):
                n_out = n.activate(self.activs, batched=True)
                self.activs[:,x.shape[1]+ix] = n_out
                if n.is_output == True:
                    out.append(n_out)
        print(self.activs)
        return torch.tensor(out)
    
    def get_weights(self, flattened=False):
        if not flattened:
            return self.get_weights_as_dict()
        else:
            return self.get_weights_flattened()

    def get_weights_flattened(self):
        weights_list = torch.tensor()
        for n in self.nodes:
            weights_list += n.weights
        return weights_list
        
    def get_weights_as_dict(self):
        weight_dict = {}
        for n in self.nodes:
            weight_dict[n.node_key] = n.weights
        return weight_dict

    def print_model_details(self):
        for node in self.nodes:
            print(f"node key {node.node_key}, connections to {node.in_idxs}")
        num_cons = sum(len(n.in_idxs) for n in self.nodes)
        print(f"{len(self.nodes)} total nodes \n {num_cons} total connections")