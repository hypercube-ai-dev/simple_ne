import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class EliteEsPop(object):
     #
    def __init__(self, net_obj:torch.nn.Module, sigma=.001, elite_cutoff=.1, size=100):
        self.sigma = sigma
        self.cutoff = int(elite_cutoff * size)
        print(self.cutoff)
        self.param_vec = parameters_to_vector(net_obj.parameters())
        self.pop = torch.randn(size, self.param_vec.shape[0])

    def evolve_pop(self, elite_idxs):
        new_pop = torch.index_select(self.pop, 0, elite_idxs)
        print(new_pop.shape)
        for i in range(self.cutoff):
            mutated = (self.sigma * torch.randn(1, self.pop.shape[1])) + new_pop[i]
            new_pop = torch.cat([new_pop, mutated], dim=0)
        new_individuals = self.pop.shape[0] - new_pop.shape[0]
        self.pop = torch.cat([new_pop, torch.randn(new_individuals, new_pop.shape[1])])