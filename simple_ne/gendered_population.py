import torch
from torch import nn
from random import random
from .activations import activations
from .attention_nets import AttentionNeNet, AttentionNeNode
from .base_population import SimpleNEPopulation

class GenderedPopulation():
    def __init__(
            self, 
            male_input_size, 
            male_output_size,
            female_input_size, 
            female_output_size, 
            max_size, 
            pop_size,  
            output_activation = None,
            female_output_activation = "sigmoid",
            prob_params = None,
            in_layer=True):
        self.male_pop = SimpleNEPopulation(
            male_input_size, 
            male_output_size, 
            max_size, 
            pop_size//2, 
            output_activation=output_activation,
            prob_params=prob_params,
            in_layer=in_layer)
        self.female_pop = SimpleNEPopulation(
            female_input_size, 
            female_output_size, 
            max_size, 
            pop_size//2, 
            output_activation=female_output_activation,
            prob_params=prob_params,
            in_layer=in_layer)
        
    def evaluate(self, inputs, female_eval, male_eval):
        male_outs = [male_eval(inputs, m_net) for m_net in self.male_pop.population]
        female_outs = []
        for ix, m_out in enumerate(male_outs):
            female_outs[ix] = [female_eval(inputs, m_out, f_net) for f_net in self.female_pop.population]
        return male_outs

    def evolve(self, fitness_list):
        self.male_pop.evolve(torch.mean(fitness_list, 1))
        self.female_pop.evolve(torch.mean(fitness_list, 0))
            
