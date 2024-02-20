from .activations import activations
from .attention_nets import AttentionNeNet, AttentionNeNode
from .base_population import SimpleNEPopulation

class SimpleNEAttentionPopulation(SimpleNEPopulation):
    def __init__(
            self, 
            input_size, 
            output_size, 
            max_size, 
            pop_size, 
            species=1, 
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