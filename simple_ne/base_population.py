import torch
from torch import nn
from random import random
from .activations import activations
from .base_networks import SimpleNEAgent, SimpleNENode

class SimpleNEPopulation():
    def __init__(
            self, 
            input_size, 
            ouput_size, 
            max_size, 
            pop_size, 
            species=1, 
            output_activation = None,
            prob_params = None,
            in_layer=True):
        self.max_nodes = max_size
        self.pop_size = pop_size
        self.in_size = input_size
        self.out_size = ouput_size
        self.output_activation = output_activation
        self.population = []
        self.in_layer = in_layer
        '''
        0 -> elitism (0,1)
        1 -> add node prob (0,1)
        2 -> mutate rate (0,1)
        3 -> prob add connection (0,1)
        '''
        if prob_params != None:
            self.prob_params = prob_params
        else:
            if species == 1:
                self.prob_params = torch.rand(4)
            else:
                self.prob_params = torch.rand(species, 4)
        self.elite_cutoff = int(self.pop_size * self.prob_params[0])
        self.init_population()

    def init_population(self):
        for i in range(self.pop_size):
            self.population.append(self.create_genome())
        return
    
    def create_genome(self):
        nodes = []
        '''
        for i in range(self.in_size):
            activation_key = torch.argmax(torch.rand(len(activations)))
            key = i
            connection_keys = (torch.rand(self.input_size) < self.prob_params[3]).nonzero()
            weights = torch.rand(len(connection_keys))
        '''
        add_node = random() < self.prob_params[1]
        if self.in_layer:
            for x in range(self.in_size):
                activation_key = torch.argmax(torch.rand(len(activations)))
                connection_keys = torch.tensor([x])
                key = self.in_size + x
                nodes.append(self.create_node(
                    activation_key,
                    connection_keys,
                    key
                ))
        if add_node:
            activation_key = torch.argmax(torch.rand(len(activations)))
            connection_keys = self.get_connection_keys(1)
            if len(connection_keys.size()) == 0:
                connection_keys = torch.tensor([torch.argmax(torch.rand(self.in_size + self.out_size + len(nodes)-2))])
            key = len(nodes)
            nodes.append(self.create_node(
                activation_key,
                connection_keys,
                key
            ))
        for i in range(self.out_size):
            if self.output_activation != None:
                activation_key = self.output_activation
            else:
                activation_key = torch.argmax(torch.rand(len(activations)))
            connection_keys = self.get_connection_keys(len(nodes))
            if len(connection_keys.size()) == 0:
                connection_keys = torch.tensor([torch.argmax(torch.rand(self.in_size + self.out_size + len(nodes)-2))])
            key = self.in_size + len(nodes) + i
            print("adding output node")
            nodes.append(self.create_node(
                activation_key,
                connection_keys,
                key,
                is_output=True
            ))
        return self.create_net(nodes, self.in_size, self.out_size)
    
    def mutate_genome(self, net: SimpleNEAgent):
        mutated = False
        mutate_probs = torch.rand(len(net.nodes))
        mutate_node_idxs = (mutate_probs < self.prob_params[2]).nonzero()
        net_nodes = net.nodes[:]
        for i in mutate_node_idxs:
            mutated = True
            n = net_nodes[i]
            mutation = torch.rand(len(n.in_idxs)) - torch.rand(len(n.in_idxs))
            n.weights += mutation
        add_node = random() < self.prob_params[1]
        if add_node == True:
            mutated = True
            activation_key = torch.argmax(torch.rand(len(activations)))
            connection_keys = self.get_connection_keys(len(net_nodes))
            if len(connection_keys.size()) == 0:
                connection_keys = torch.tensor([torch.argmax(torch.rand(self.in_size + self.out_size + len(net_nodes)-2))])
            weights = torch.randn(len(connection_keys))
            key = len(net_nodes)
            net_nodes.append(self.create_node(
                activation_key,
                connection_keys,
                key,
                weights=weights
            ))
            add_conns_from_node = (torch.rand(len(net_nodes) - 1) < self.prob_params[3]).nonzero()
            for i in add_conns_from_node:
                to_node = net_nodes[i]
                torch.cat((to_node.in_idxs, torch.tensor([key])))
                torch.cat((to_node.weights, torch.randn(1)))
        if mutated == True:
            return self.create_net(net_nodes, self.in_size, self.out_size)
        else:
            return None

    def cross_over(self, net_one, net_two):
        return
    
    def get_connection_keys(self, num_nodes):
        # all nodes should have the ability to connect with any other node
        return (torch.rand(self.in_size + self.out_size + num_nodes-2) < self.prob_params[3]).nonzero().squeeze()
    
    def evolve(self, fitness_list):
        top_nets, top_net_idxs = torch.topk(fitness_list, self.elite_cutoff)
        elites = [self.population[i] for i in top_net_idxs]
        self.population = []
        for i in range(len(elites)):
            n = elites[i]
            self.population.append(n)
            if i < len(elites) //2:
                mutated = self.mutate_genome(n)
                if (mutated != None):
                    self.population.append(mutated)
        for i in range (self.pop_size - len(self.population)):
            self.population.append(self.create_genome())

    def create_node(self, activation_ix, connections, node_key, weights=None, is_output=False):
        return SimpleNENode(
            activations[activation_ix],
            connections,
            node_key,
            weights=weights)
    
    def create_net(self, nodes, in_size, out_size):
        return SimpleNEAgent(nodes, in_size, out_size)