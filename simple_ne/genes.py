class NodeGene:
    def __init__(self, node_id, node_type, activation='relu'):
        self.id = node_id
        self.type = node_type  # 'input', 'hidden', 'output'
        self.activation = activation

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled