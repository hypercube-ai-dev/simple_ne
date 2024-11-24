import torch
from torch import nn
from random import random
from .activations import activations

class SimpleNENode(object):
    def __init__(self, activation, in_idxs, key, weights=None, is_output=False, agg_func=None):
        self.activation = activation
        if weights == None:
            self.weights = torch.randn(len(in_idxs))
        else:
            self.weights = weights
        self.in_idxs = in_idxs
        self.node_key = key
        self.is_output = is_output
        self.agg = agg_func

    def activate(self, inputs, batched=False):
        if batched == True:
            inputs = torch.index_select(inputs, 1, self.in_idxs)
        else:
            inputs = torch.index_select(inputs, 0, self.in_idxs)
        if self.agg != None:
            inputs = self.agg(inputs)
        return self.activation(torch.matmul(inputs, self.weights))
    
    def add_connection(self, key):
        torch.cat((self.in_idxs, torch.tensor([key])))
        torch.cat((self.weights, torch.randn(1)))

class SimpleNEAgent(nn.Module):
    def __init__(self, nodes: list[SimpleNENode], input_size, output_size):
        super().__init__()
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.in_size = input_size
        self.out_size = output_size
        return

    # nodes are added in order and have a
    # probability of connecting to existing nodes
    # each node will need the output of nodes that come before it
    def forward(self, x, mask=None):
        self.activs = torch.zeros(x.shape[0], x.shape[1] + self.num_nodes)
        print(len(x.shape))
        if len(x.shape) == 1:
            print(self.activs.shape)
            print(x.shape)
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
        if self.out_size > 1:
            return torch.tensor(out)
        else:
            return out[0]
    
    def get_weights(self, flattened=False):
        if not flattened:
            return self.__get_weights_as_dict()
        else:
            return self.__get_weights_flattened()

    def __get_weights_flattened(self):
        weights_list = []
        for n in self.nodes:
            weights_list += n.weights
        return torch.tensor(weights_list)
        
    def __get_weights_as_dict(self):
        weight_dict = {}
        for n in self.nodes:
            weight_dict[n.node_key] = n.weights
        return weight_dict

    def print_model_details(self):
        for node in self.nodes:
            if node.is_output:
                print("output node")
            print(f"node key {node.node_key}, connections to {node.in_idxs}")
        num_cons = sum(len(n.in_idxs) for n in self.nodes)
        print(f"{len(self.nodes)} total nodes \n {num_cons} total connections")