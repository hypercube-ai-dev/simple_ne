from simple_ne.simple_ne import SimpleNEAgent, SimpleNENode, SimpleNEPopulation
from simple_ne.sub_cube import SubDivisionCube
import torch
from simple_ne.hypercube_helper import HypercubeHelper
import simple_ne.hypercube_attention


cube = SubDivisionCube((0.0,0.0,0.0), 2, .5)
bp = SimpleNEPopulation(2, 2, 100, 10)

def get_weights_tests():
    test_genome = bp.create_genome()
    print(test_genome.get_weights())
    print(test_genome.get_weights(True))

get_weights_tests()

def tree_test():
    print(cube.tree[0].shape)
    print(len(cube.tree[0][1]))

def test_encode_input_layer(num_inputs=2):
    test_genome = bp.create_genome()
    input_substrate = [[.5, .5, .5/x] for x in range(num_inputs)]
    

tree_test()
