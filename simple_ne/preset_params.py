import torch

def bernolli(num_params):
    return torch.full([num_params], .5)

def random_params(num_params):
    return torch.rand(num_params)

str_to_param = {
    'bernolli': bernolli,
    'random': random_params
}

def get_named_params(param_name, num_params):
    return str_to_param[param_name](num_params)