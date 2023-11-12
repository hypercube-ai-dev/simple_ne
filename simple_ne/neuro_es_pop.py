import torch
from torch import nn
from random import random

from simple_ne.base_networks import SimpleNEAgent
from .activations import activations
from .base_networks import SimpleNEAgent, SimpleNENode
from .attention_nets import AttentionNeNet, AttentionNeNode
from .base_population import SimpleNEPopulation

class SimpleNeEsPopulation(SimpleNEPopulation):
    def __init__(
            self, 
            input_size, 
            output_size, 
            max_size, 
            pop_size=100, 
            species=10, 
            output_activation = None,
            prob_params = None,
            in_layer=True):
        super().__init__(
            input_size,
            output_size,
            max_size,
            pop_size,
            species,
            output_activation,
            prob_params,
            in_layer
        )
        self.num_species = species
    
    def set_prob_params(self, prob_params, species):
        if prob_params != None:
            self.prob_params = prob_params
        else:
            self.prob_params = torch.rand(4)

    def init_population(self):
        for i in range(self.num_species):
            # will need to use modulo to determine which species in
            # reproduction/mutation logic
            initial_genome = self.create_genome()
            weight_dict = initial_genome.get_weights_as_dict()
            self.population.append(initial_genome)
            for g in range((self.pop_size // self.num_species)-1):
                es_g_nodes = []
                for n in initial_genome.nodes:
                    new_weights = torch.randn(n.weights.shape)
                    es_g_nodes.append(SimpleNENode(n.activation, n.in_idxs, n.node_key, new_weights))
                self.population.append(SimpleNEAgent(es_g_nodes, initial_genome.in_size, initial_genome.out_size, initial_genome.batch_size))
        return
    
    def mutate_genome(self, net: SimpleNEAgent):
        net_weights = net.get_weights_as_dict()
        return

class SimpleNeEsAttentionPopulation(SimpleNeEsPopulation):
    def __init__(
            self, 
            input_size, 
            output_size, 
            max_size, 
            pop_size, 
            species=2, 
            output_activation = None,
            prob_params = None,
            max_context_len = 8,
            in_layer=True):
        super().__init__(
            input_size,
            output_size,
            max_size,
            pop_size,
            species,
            output_activation,
            prob_params,
            in_layer
        )
        self.max_context_len = max_context_len
    
    def create_node(self, activation_ix, connections, node_key, is_output=False):
        return AttentionNeNode(
            connections, 
            activations[activation_ix],
            node_key,
            is_output=is_output)
    
    def create_net(self, nodes, in_size, out_size):
        return AttentionNeNet(
            nodes,
            in_size, 
            out_size,
            self.max_context_len
        )