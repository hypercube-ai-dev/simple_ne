from sub_cube import SubDivisionCube

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