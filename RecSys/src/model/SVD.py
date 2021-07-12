import sys

import torch as th
from torch import nn
import numpy as np


class IntegratedSVD(nn.Module):

    def __init__(self, data, n_user, n_item, h_dim):

        super(IntegratedSVD, self).__init__()


        self.n_user = n_user
        self.n_item = n_item

        self.mu = th.mean(data)

        self.b_u = nn.Parameter(th.rand(m))
        self.b_i = nn.Parameter(th.rand(n))

        self.P = nn.Parameter(th.rand(m, hid_dim))
        self.Q = nn.Parameter(th.rand(hid_dim, n))

        self.c = nn.Parameter(th.rand(m, n))


    def forward(self):

        b_ui = self.mu + self.b_u + self.b_i

        NBD_model = self.Q.transpose(1, 0) * (self.P)

        # LF_model =
        # r_ui = b_ui + th.sqrt(|R_u|) * \th.sum(r_uj - b_uj)*w_ij + th.sqrt(|N(u)|)*th.sum(c_ij)