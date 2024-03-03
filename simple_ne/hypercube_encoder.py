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
    
    def encode_input_layer(self, in_coords):
        out_coords = self.cube.tree[0]
        return 