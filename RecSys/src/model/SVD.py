import sys

import torch as th
from torch import nn
from torch.optim import Adam
import numpy as np


class IntegratedSVD(nn.Module):

    def __init__(self, n_user, n_item, mu, bu, bi, R, h_dim):

        super(IntegratedSVD, self).__init__()

        self.mu = mu
        self.bu = bu
        self.bi = bi
        self.R = R

        self.w_ij = nn.Parameter(th.rand(n_user, n_item))
        self.P = nn.Parameter(th.rand(n_user, h_dim))
        self.Q = nn.Parameter(th.rand(n_item, h_dim))

        self.optimizer = Adam(self.parameters(), lr=1e-3)

        # No Implicit Feedback
        # self.c = nn.Parameter(th.rand(m, n))


    def forward(self, u, i, r):

        b_ui = self.mu + self.b_u + self.b_i

        pred = b_ui + self.Q[i].tranpose(1, 2)(self.P[u] + len(self.R_k[u, i])*th.sum(r-b_ui)*self.w_ij)

        return pred


    def train_batch(self, u, i, r, train=True):

        pred = self.forward(u, i, r)

        loss = nn.RMSELoss()(pred, r)

        if train:
            self.optimizer.set_zero_grad
            loss.backward()
            self.optimizer.step()

        return loss