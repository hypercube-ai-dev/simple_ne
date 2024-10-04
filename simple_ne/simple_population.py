from genome import Genome
from genes import NodeGene, ConnectionGene
import numpy as np
import torch.nn as nn

class SimplePopulation():
    def __init__(self, in_size, out_size, size=10):
        self.size = size
        self.in_size = in_size
        self.out_size = out_size
        self.population = []

    def create_initial_population(self):
        for _ in range(self.size):
            genome = Genome()
            # Add input nodes
            for i in range(self.in_size):
                genome.add_node(NodeGene(i, 'input'))
            # Add output nodes
            for i in range(self.in_size, self.in_size + self.out_size):
                genome.add_node(NodeGene(i, 'output', activation='sigmoid'))
            # Fully connect input and output nodes
            for in_node in range(self.in_size):
                for out_node in range(self.in_size, self.in_size + self.out_size):
                    weight = np.random.uniform(-1, 1)
                    genome.add_connection(ConnectionGene(in_node, out_node, weight))
            self.population.append(genome)