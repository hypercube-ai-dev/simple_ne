from simple_ne.simple_ne import SimpleNEAgent, SimpleNENode, SimpleNEPopulation
from simple_ne.sub_cube import SubDivisionCube
import torch
import simple_ne.hypercube_attention as ha


def get_weights_tests():
    bp = SimpleNEPopulation(6, 1, 100, 10)
    test_genome = bp.create_genome()
    print(test_genome.get_weights())
    print(test_genome.get_weights(True))

get_weights_tests()

def tree_test():
    cube = SubDivisionCube((0.0,0.0,0.0), 2, .5)
    print(cube.tree[0].shape)
    print(len(cube.tree[0][1]))

def test_encode_input_layer(num_inputs=20):
    cube = SubDivisionCube((0.5,0.5,0.5), 2, .5)
    # input six for 3d substrate, out one for weight
    bp = SimpleNEPopulation(6, 1, 100, 10)
    test_genome = bp.create_genome()
    input_substrate = torch.tensor([[1.0, 1.0, 1.0/x] for x in range(1,num_inputs+1)])
    print(input_substrate)
    merged = ha.get_nd_coord_inputs_as_tensor(input_substrate, cube.tree[0])
    out = test_genome(merged)
    print(out)
    
test_encode_input_layer()
