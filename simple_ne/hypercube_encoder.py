from sub_cube import SubDivisionCube
import torch as pt

# REMINDER: we will want float32 type outputs as that
# is what pytorch uses as default


class HypercubeEncoder():

    def __init__(
            self,
            cppn,
            initial_depth,
            center,
            width
    ):
        self.cube = SubDivisionCube(center, initial_depth, width)
        self.cppn = cppn
    
    def encode_input_layer(self, in_coords, net):
        return net(in_coords, self.cube.tree[0])

    # use this to encode weights between any two depths of the subdivision tree
    # these can be the same depth if desired
    def encode_hiddens(self, from_depth, to_depth, net):
        return net(self.cube.tree[from_depth], self.cube.tree[to_depth])

    def encode_output_layer(self, from_depth, out_coords, net):
        return net(self.cube.tree[from_depth], out_coords)