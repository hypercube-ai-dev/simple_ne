import torch
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class AttentionNeNode(torch.nn.Module):

    def __init__(self, in_keys, activation, node_key, is_output=False):
        super().__init__()
        self.in_idxs = in_keys
        # qkv so multiply by three
        self.weights = torch.randn(len(in_keys), 3)
        #self.o_proj = torch.randn(len(in_keys), 1)
        self.activation = activation
        self.key = node_key
        self.is_output = is_output

    def forward(self, inputs, batched=False):
        if batched == True:
            inputs = torch.index_select(inputs, 2, self.in_idxs)
        else:
            inputs = torch.index_select(inputs, 1, self.in_idxs)
        qkv_proj = torch.matmul(inputs, self.weights)
        q,k,v = qkv_proj.chunk(3, dim=-1)
        values,_ = scaled_dot_product(q,k,v)
        out = self.activation(values[-1:,])
        #out = self.activation(torch.matmul(values[-1:,], self.o_proj))
        return out
    
    def add_connection(self, key):
        torch.cat((self.in_idxs, torch.tensor([key])))
        # add qkv weights for new incoming connection
        torch.cat((self.weights, torch.randn(1, 3)))

class AttentionNeNet(torch.nn.Module):

    def __init__(self, nodes: AttentionNeNode, in_size, out_size, max_context_length):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.nodes = nodes
        self.max_length = max_context_length
        self.no_actives = True
        
    def reset(self, batch_size=0):
        if batch_size == 0:
            self.no_actives = True
            self.actives = torch.zeros(1, self.in_size + len(self.nodes) + self.out_size)
        else:
            self.no_actives = True
            self.actives = torch.zeros(batch_size, 1, self.in_size + len(self.nodes) + self.out_size)

    def forward(self, inputs, batched=False):
        if not batched:
            return self.activate(inputs)
        else:
            return self.activate_batched(inputs)
    
    def activate(self, x):
        if self.no_actives == False:
            new_activs = torch.zeros(1,self.actives.shape[1])
            new_activs[:,:x.shape[0]] = x
            if self.actives.shape[0] == self.max_context_length:
                self.actives = self.actives[1:,]
            self.actives = torch.cat((self.actives, new_activs), dim=0)
        else:
            self.actives[:,:x.shape[0]] = x
        out = []
        for ix,n in enumerate(self.nodes):
            n_out = n(self.actives)
            self.actives[-1:,x.shape[0]+ix] = n_out
            if n.is_output == True:
                out.append(n_out)
        return torch.tensor(out)
    
    def activate_batched(self, inputs):
        return