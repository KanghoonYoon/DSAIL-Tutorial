import sys
import os

import pandas as pd
import torch as th

class NetflixData(object):

    def __init__(self, u, i, r):

        self.u = u
        self.i = i
        self.r = r

    def __getitem__(self, idx):

        return self.u[idx], self.i[idx], self.r[idx]

def read_data(dataname):

    u = []
    i = []
    r = []

    if dataname == 'netflix':
        path = 'data/Netflix/training_set'

        for file in os.listdir(path):

            with open(path+'/'+file, 'r') as f:


                item_idx = int(file.replace('mv_', '').replace('.txt', ''))

                print("Column names:", f.readline())

                while 1:
                    line = f.readline()


                    if line == '':
                        break

                    line = line.replace('\n', '')
                    user_idx, rating, data = line.split(',')
                    user_idx = int(user_idx)
                    rating = int(rating)
                    u.append(user_idx)
                    i.append(item_idx)
                    r.append(rating)

    return u, i, r


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def collate_fn(instance):

    u, i, r = instance

    return th.FloatTensor(u), th.FloatTensor(i), th.FloatTensor(r)
