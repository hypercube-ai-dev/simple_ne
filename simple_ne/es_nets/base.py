import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class EsBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    def param_vector(self):
        return parameters_to_vector(self.parameters())
    
    def set_params(self, new_params):
        vector_to_parameters(new_params, self.parameters())
        return self
