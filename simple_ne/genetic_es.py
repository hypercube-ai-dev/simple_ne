import torch
from torch import nn
from random import random

from simple_ne.base_networks import SimpleNEAgent
from .activations import activations
from .base_networks import SimpleNEAgent, SimpleNENode
from .attention_nets import AttentionNeNet, AttentionNeNode
from .base_population import SimpleNEPopulation
from .simple_es.cma_es_torch import CMAES

'''
This class implements a hybrid neuroevolution/es population
neuroevolution is used to discover new topologies, by generating
species_size number of networks, the weights of these networks are 
optimized with an evolution strategy for es_epochs iterations, after
the top 1/3 species will be preserved and mutated to account for 2/3 of 
the next species size while the final 1/3 of this new population of species will
be brand new
'''

class GeneticEsPop(SimpleNEPopulation):
    def __init__(
            self, 
            input_size, 
            output_size, 
            max_size, 
            pop_size=100, 
            species=10, 
            output_activation = None,
            prob_params = None,
            in_layer=True,
            es_epochs=10):
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
        self.species_size = (pop_size // species) - 1
        self.sigma = .1

    def init_population(self):
        self.pop = []
        for i in range(self.num_species):
            # will need to use modulo to determine which species in
            # reproduction/mutation logic
            self.pop.append(self.create_genome())
        return

    def evolve(self, fitness_func):
        #TODO parallize this so that each species runs es
        # in its own thread
        for s in self.pop:
            weights = s.get_weights()
            es = CMAES(len(weights), self.species_size)
            es.mean = weights
            es_pop = es.ask()
            fits = sorted([(nw, fitness_func(s.set_weights(nw))) for nw in es_pop], key=lambda x:x[1])

            
    def __es_mutate(self, initial_genome, survivor):
        es_g_nodes = []
        for n in initial_genome.nodes:
            if survivor:
                new_weights = n.weights + (self.sigma * torch.rand(n.weights.shape))
            new_weights = torch.randn(n.weights.shape)
            es_g_nodes.append(SimpleNENode(n.activation, n.in_idxs, n.node_key, new_weights, n.is_output))
        return SimpleNEAgent(es_g_nodes, initial_genome.in_size, initial_genome.out_size, initial_genome.batch_size)