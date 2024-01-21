import torch
import itertools

class SubDivisionCube(object):
    def __init__(self, center, depth, width):
        self.width = width
        self.dim = len(center)
        self.center = torch.tensor(center)
        self.init_depth = depth
        self.signs = torch.tensor(list(itertools.product([width,-width], repeat=self.dim)))
        self.tree = []
        self.sub_cube_size = (2**self.dim)
        self.build_tree()

    def build_tree(self):
        depth = 0
        while depth < self.init_depth:
            depth += 1
            if depth == 1:
                self.tree.append(self.center.repeat(self.sub_cube_size, 1) + (self.signs / (2*depth)))
            else:
                print(self.tree[depth-2])
                # repeat depth minus one so that we match previous 
                offsets = (self.signs / (2*depth)).repeat(self.sub_cube_size ** (depth - 1), 1)
                self.tree.append(torch.repeat_interleave(self.tree[depth-2],self.sub_cube_size,0) + offsets)
        return


class BatchednDimensionTree:
    
    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.child_coords = []
        self.cs = []
        self.signs = self.set_signs() 

    def set_signs(self):
        return list(itertools.product([1,-1], repeat=len(self.coord)))
    
    def divide_childrens(self):
        for x in range(self.num_children):
            new_coord = []
            for y in range(len(self.coord)):
                new_coord.append(self.coord[y] + (self.width/(2*self.signs[x][y])))
            self.child_coords.append(new_coord)
            newby = BatchednDimensionTree(new_coord, self.width/2, self.lvl+1)
            self.cs.append(newby)