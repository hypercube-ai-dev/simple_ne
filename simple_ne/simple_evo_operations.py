from genes import ConnectionGene, NodeGene
import random
import numpy as np

def add_node_mutation(genome):
    # Select a connection to split
    connection = random.choice(genome.connections)
    connection.enabled = False
    new_node_id = max(genome.nodes.keys()) + 1
    new_node = NodeGene(new_node_id, 'hidden', activation=random.choice(['relu', 'tanh', 'sigmoid']))
    genome.add_node(new_node)
    # Add new connections
    genome.add_connection(ConnectionGene(connection.in_node, new_node_id, weight=1.0))
    genome.add_connection(ConnectionGene(new_node_id, connection.out_node, weight=connection.weight))

def add_connection_mutation(genome):
    possible_nodes = list(genome.nodes.keys())
    in_node = random.choice(possible_nodes)
    out_node = random.choice(possible_nodes)
    if in_node == out_node:
        return
    # Check if connection already exists
    if genome.connection_exists(in_node, out_node):
        return
    weight = np.random.uniform(-1, 1)
    genome.add_connection(ConnectionGene(in_node, out_node, weight))

def weight_mutation(genome):
    for connection in genome.connections:
        if random.random() < 0.8:
            connection.weight += np.random.normal(0, 0.1)
        else:
            connection.weight = np.random.uniform(-1, 1)
            
def crossover(parent1, parent2):
    child = Genome()
    # Combine nodes
    all_nodes = {**parent1.nodes, **parent2.nodes}
    child.nodes = copy.deepcopy(all_nodes)
    # Combine connections
    all_connections = parent1.connections + parent2.connections
    # Remove duplicates
    unique_connections = {(conn.in_node, conn.out_node): conn for conn in all_connections}
    child.connections = list(unique_connections.values())
    return child
