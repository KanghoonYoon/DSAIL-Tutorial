from argparse import ArgumentParser

import torch as th
from torch import nn
from torch.optim import Adam

import dgl
from dgl import DGLGraph

from src.nn.GAT import GraphAttentionLayer
from src.nn.GAT import MultiheadGraphAttentionLayer

class GATClassifier(nn.Module):

    def __init__(self, input_dim, h_dim, out_dim, n_heads):

        super(GATClassifier, self).__init__()

        self.input_layer = nn.Linear(input_dim, h_dim)
        self.gat_layer1 = MultiheadGraphAttentionLayer(h_dim, h_dim, n_heads)
        self.gat_layer2 = MultiheadGraphAttentionLayer(h_dim, h_dim, n_heads)
        self.output_layer = nn.Linear(h_dim, out_dim)

        self.optimizer = Adam(self.parameters(), lr=1e-3)

    def forward(self, g, x):

        h = self.input_layer(x)
        h = nn.ReLU()(self.gat_layer1(g, h))
        h = nn.ReLU()(self.gat_layer2(g, h))
        o = self.output_layer(h)

        return o

    def train_batch(self, g, x, y, mask, train):

        pred = self.forward(g, x)

        loss = nn.CrossEntropyLoss()(pred[mask], y[mask])

        self.optimizer.zero_grad()

        if train:
            loss.backward()
            self.optimizer.step()

        return loss, th.argmax(pred[mask], dim=1)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default='cora')
    parser.add_argument("--h_dim", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--prt_evry", type=int, default=10)

    config = parser.parse_args()

    config.device = 'cuda' if th.cuda.is_available() else 'cpu'


    if config.data == "cora":
        dataset = dgl.data.CoraGraphDataset()
    elif config.data == "citeseer":
        dataset = dgl.data.CiteseerGraphDataset()
    elif config.data == "pubmed":
        dataset = dgl.data.PubmedGraphDataset()


    graph = DGLGraph(dataset.graph).to(config.device)
    X = dataset.features.to(config.device)
    y = th.LongTensor(dataset.labels).to(config.device)
    train_mask = th.FloatTensor(dataset.train_mask).bool().to(config.device)
    val_mask = th.FloatTensor(dataset.val_mask).bool().to(config.device)

    model = GATClassifier(input_dim=X.size(1), h_dim=config.h_dim, out_dim=dataset.num_classes, n_heads=config.n_heads)
    model.to(config.device)


    for epoch in range(config.epochs):

        train_loss, train_pred = model.train_batch(graph, X, y, train_mask, True)
        train_acc = th.sum(y[train_mask]==train_pred).float()/th.sum(train_mask)
        val_loss, val_pred = model.train_batch(graph, X, y, val_mask, False)
        val_acc = th.sum(y[val_mask] == val_pred).float() / th.sum(val_mask)

        if epoch%config.prt_evry ==0:
            print("Train Loss:{}, Train Accuracy:{}".format(train_loss, train_acc))
            print("Val Loss:{}, Val Accuracy:{}".format(val_loss, val_acc))

    print("END")

