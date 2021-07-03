import torch as th
from torch import nn
from torch.nn import functional as F

import dgl
from dgl import function as fn
from dgl.ops import edge_softmax

class GraphAttentionLayer(nn.Module):

    def __init__(self, input_dim, h_dim):

        super(GraphAttentionLayer, self).__init__()

        self.W_e = nn.Linear(2*input_dim, h_dim)


    def forward(self, g, h):

        with g.local_scope():

            g.ndata['h'] = h
            g.apply_edges(self.message_func)

            e = g.edata['e']
            g.edata['alpha'] = edge_softmax(g, e)
            g.update_all(message_func=fn.src_mul_edge('h', 'alpha', 'm'), reduce_func=fn.sum('m', 'h'))

            h = g.ndata['h']

        return h

    def message_func(self, edges):
        x = th.cat([edges.src['h'], edges.dst['h']], dim=-1) ## Check
        e = F.leaky_relu(self.W_e(x))
        return {'e': e}


class MultiheadGraphAttentionLayer(nn.Module):

    def __init__(self, input_dim, h_dim, n_heads):

        super(MultiheadGraphAttentionLayer, self).__init__()

        self.multi_attn = nn.ModuleList([GraphAttentionLayer(input_dim, h_dim) for _ in range(n_heads)])


    def forward(self, g, h):

        hs = []

        for attn_layer in self.multi_attn:

            hs.append(attn_layer(g, h))

        # hs = th.cat(hs, dim=1)
        hs = th.stack(hs, -1)
        hs = th.mean(hs, -1)

        return hs

    def message_func(self, edges):
        x = th.cat([edges.src['h'], edges.dst['h']], dim=-1) ## Check
        e = F.leaky_relu(self.W_e(x))
        return {'e': e}
