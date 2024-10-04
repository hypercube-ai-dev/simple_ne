import torch.nn as nn
import torch

class Genome:
    def __init__(self):
        self.nodes = {}  # {node_id: NodeGene}
        self.connections = []  # List of ConnectionGene

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_connection(self, connection):
        self.connections.append(connection)

    def connection_exists(self, in_node, out_node):
        for conn in self.connections:
            if conn.in_node == in_node and conn.out_node == out_node:
                return True
        return False

    def build_network(self):
        return DynamicNet(self)

class DynamicNet(nn.Module):
    def __init__(self, genome):
        super(DynamicNet, self).__init__()
        self.genome = genome

    def forward(self, x):
        node_values = {}
        # Assign input values
        for node in self.genome.nodes.values():
            if node.type == 'input':
                node_values[node.id] = x[:, node.id].view(-1, 1)

        # Process nodes in topological order
        processing_order = self.topological_sort()
        for node_id in processing_order:
            node = self.genome.nodes[node_id]
            if node.type == 'input':
                continue
            inputs = []
            for conn in self.genome.connections:
                if conn.enabled and conn.out_node == node_id:
                    in_value = node_values.get(conn.in_node)
                    if in_value is not None:
                        inputs.append(in_value * conn.weight)
            if inputs:
                total_input = sum(inputs)
                # Apply activation function
                activation = getattr(F, node.activation)
                node_values[node_id] = activation(total_input)
        # Collect outputs
        outputs = [node_values[node.id] for node in self.genome.nodes.values() if node.type == 'output']
        return torch.cat(outputs, dim=1)

    def topological_sort(self):
        # Implement topological sort to handle arbitrary acyclic graphs
        visited = set()
        order = []

        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for conn in self.genome.connections:
                if conn.enabled and conn.in_node == node_id:
                    dfs(conn.out_node)
            order.append(node_id)

        for node_id in self.genome.nodes:
            dfs(node_id)
        return reversed(order)
