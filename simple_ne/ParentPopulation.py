from torch import nn
from random import random
from .activations import activations
from .attention_nets import AttentionNeNet, AttentionNeNode
from .base_population import SimpleNEPopulation

class ParentPopulation():
    def __init__(
            self, 
            male_input_size, 
            male_output_size,
            female_input_size, 
            female_output_size, 
            max_size, 
            pop_size,  
            output_activation = None,
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
            output_activation=output_activation,
            prob_params=prob_params,
            in_layer=in_layer)