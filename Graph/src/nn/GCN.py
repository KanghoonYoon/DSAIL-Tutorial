import torch as th
from torch import nn

import dgl

from dgl import function as fn



class GraphConvolutionalLayer(nn.Module):

    def __init__(self, input_dim, h_dim):

        super(GraphConvolutionalLayer, self).__init__()

        self.linear = nn.Linear(input_dim, h_dim)


    def forward(self, g, h):

        src = fn.copy_u('h', 'm')
        reduce = fn.sum('m', 'h')

        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(src, reduce) ## Check This function already includes W or not
            h = g.ndata['h']

        return self.linear(h)