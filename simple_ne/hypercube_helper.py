from sub_cube import SubDivisionCube
import torch

# REMINDER: we will want float32 type outputs as that
# is what pytorch uses as default


class HypercubeHelper():

    def __init__(
            self,
            initial_depth,
            center,
            width,
            substrate
    ):
        self.cube = SubDivisionCube(center, initial_depth, width)
        self.reset_substrate(substrate)
    
    def encode_input_layer(self, net):
        merged = get_nd_coord_inputs_as_tensor(self.substrate["input_coords"], self.cube.tree[0])
        return net(merged)

    def reset_substrate(self, new_sub):
        for k in new_sub:
            new_sub[k] = torch.tensor(new_sub[k])
        self.substrate = new_sub

    # use this to encode weights between any two depths of the subdivision tree
    # these can be the same depth if desired
    def encode_hiddens(self, from_depth, to_depth, net):
        merged = get_nd_coord_inputs_as_tensor(self.cube[from_depth], self.cube[to_depth])
        return net(merged)

    def encode_output_layer(self, from_depth, net):
        return net(self.cube.tree[from_depth], self.substrate["output_coords"])
    
def get_nd_coord_inputs_as_tensor(in_coords : torch.tensor, out_coords : torch.tensor):
    in_expanded = in_coords.unsqueeze(1)
    out_expanded = out_coords.unsqueeze(0)
    combined = torch.cat((in_expanded.expand(-1, out_coords.shape[0], -1), out_expanded.expand(in_coords.shape[0], -1, -1)), dim=2)
    return combined.view(-1, in_coords.shape[-1] * 2)