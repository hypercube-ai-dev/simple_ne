from simple_ne.simple_ne import SimpleNEAgent, SimpleNENode, SimpleNEPopulation
from simple_ne.sub_cube import SubDivisionCube
import torch

def get_weights_tests():
    bp = SimpleNEPopulation(2, 2, 100, 10)
    test_genome = bp.create_genome()
    print(test_genome.get_weights())
    print(test_genome.get_weights(True))

get_weights_tests()

def tree_test():
    cube = SubDivisionCube((0.0,0.0,0.0), 2, .5)
    print(cube.tree[0].shape)
    print(len(cube.tree[0][1]))

tree_test()
