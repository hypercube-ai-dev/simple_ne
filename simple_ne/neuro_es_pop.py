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
            output_activation,
            prob_params,
            in_layer
        )
        self.num_species = species
        self.es_size = species
        self.elite_cutoff = int(self.pop_size * self.prob_params[0])
        self.sigma = .1
    def init_population(self):
        for i in range(self.pop_size):
            # will need to use modulo to determine which species in
            # reproduction/mutation logic
            self.init_species()
        return
    
    def init_species(self, ng = None):
        survivor = False
        if ng == None:
            new_genome = self.create_genome()
        else:
            new_genome = ng
            survivor = True
        self.population.append(new_genome)
        for x in range(self.es_size):
            mg = self.__es_mutate(new_genome, survivor)
            self.population.append(mg)

    def evolve(self, fitness_list):
        top_nets, top_net_idxs = torch.topk(fitness_list, self.elite_cutoff)
        elites = [self.population[i] for i in top_net_idxs]
        self.population = []
        mutates = 0
        for x in range(len(elites)):
            self.init_species(elites[x])
            mutated = self.mutate_genome(elites[x])
            if mutated != None:
                self.init_species(mutated)
                mutates += 1
        for _ in range(self.pop_size - (len(elites) + mutates)):
            self.init_species()
            
    def __es_mutate(self, initial_genome, survivor):
        es_g_nodes = []
        for n in initial_genome.nodes:
            if survivor:
                new_weights = n.weights + (self.sigma * torch.rand(n.weights.shape))
            new_weights = torch.randn(n.weights.shape)
            es_g_nodes.append(SimpleNENode(n.activation, n.in_idxs, n.node_key, new_weights, n.is_output))
        return SimpleNEAgent(es_g_nodes, initial_genome.in_size, initial_genome.out_size, initial_genome.batch_size)
    
class SimpleNeEsParamsPopulation(SimpleNEPopulation):
    def __init__(
            self, 
            input_size, 
            output_size, 
            max_size, 
            pop_size, 
            num_species=2, 
            output_activation = None,
            prob_params = None,
            in_layer=True
    ):
        super().__init__(
            input_size,
            output_size,
            max_size,
            pop_size,
            output_activation,
            prob_params,
            in_layer
        )
        self.populations = [[] for _ in range(num_species)]
        self.num_species = num_species

    def set_prob_params(self, prob_params):
        if prob_params != None:
            self.prob_params = prob_params
        else:
            self.prob_params = torch.rand(self.num_species, 4)


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