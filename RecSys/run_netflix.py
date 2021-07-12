import sys
import random

import torch as th
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser

from src.data_utils import read_data, batch, NetflixData, collate_fn
from src.model.SVD import IntegratedSVD




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, default='netflix')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=50)

    config = parser.parse_args()
    config.device = 'cuda' if th.cuda.is_available() else 'cpu'
    n_user = 2649429
    n_item = 17770

    # 전체 DATA , mu,
    # movie set -> batch
    # item -> M x N Parameter Space

    # model 선언 model( len(user), len(item))

    # batch -> forward -> r ui 계산, len(movie) 의 스코어.

    item_list = list(range(1, n_item+1))
    u, i, r, mu, = read_data(config.data)
    data = NetflixData(u, i, r)
    trainlaoder = DataLoader(data, shuffle=True, collate_fn=collate_fn)
    model = IntegratedSVD(batch, n_user, n_item, config.h_dim)
    model.to(config.device)

    for epoch in range(1, config.epochs+1):

        random.shuffle(item_list)

        for batch_item_idx in batch(item_list):

            batch_data = []

            for item_idx in batch_item_idx:

                batch_data = batch_data + data










