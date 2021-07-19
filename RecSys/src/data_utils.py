import numpy as np
import pickle
import torch as th
from torch.utils.data import Dataset

class NetflixData(Dataset):

    def __init__(self, u, i, r):

        self.length = len(u)

        self.u = u
        self.i = i
        self.r = r

        self.mu = np.mean(r)
        self.bu = {_u:0 for _u in u}
        self.bi = {_i:0 for _i in i}
        self.R = {_u:0 for _u in u}  # user_cnt
        self.I = {_i: 0 for _i in i}  # item_cnt


        for ith in range(self.length):
            # Bias values
            # # mu : scalar
            # bu, bi : dict user->mean
            # R : user dependent : Dict -> cnt
            self.bu[u[ith]] += (r[ith]-self.mu)
            self.bi[i[ith]] += (r[ith]-self.mu)
            self.R[u[ith]] += 1
            self.I[i[ith]] += 1

        for _u in self.bu.keys():
            self.bu[_u] /= self.R[_u]

        for _i in self.bi.keys():
            self.bi[_i] /= self.I[_i]

        self.bu = [_bu for _u, _bu in sorted(self.bu, key=lambda x:x[0])]
        self.bi = [_bi for _i, _bi in sorted(self.bi, key=lambda x:x[0])]
        self.R = [_r for _i, _r in sorted(self.R, key=lambda x:x[0])]

    def __getitem__(self, idx):

        return self.u[idx], self.i[idx], self.r[idx]

def read_data(dataname):

    if dataname=='netflix_toy':
        with open('data/Netflix/netflix_toy.pkl', 'rb') as f:
            data = pickle.load(f)

    return data['user'], data['item'], data['rate']


def collate_fn(instance):

    u, i, r = instance

    return th.FloatTensor(u), th.FloatTensor(i), th.FloatTensor(r)
