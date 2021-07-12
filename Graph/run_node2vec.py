"""
The dataset is available from the following links,

    youtube: https://data.dgl.ai/dataset/DeepWalk/youtube.zip
    blog: https://data.dgl.ai/dataset/DeepWalk/blog.zip

"""


from argparse import ArgumentParser

import torch as th
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import dgl
from dgl import DGLGraph

from src.data_utils import Node2vecDataset


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default='blog')
    parser.add_argument("--h_dim", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--prt_evry", type=int, default=10)

    config = parser.parse_args()

    config.device = 'cuda' if th.cuda.is_available() else 'cpu'


    data = DeepWalkDataset(config.data)

    # train_loader = D
    # ataLoader(dataset=data.sample_randomwalk() )




