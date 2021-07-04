import time

import dgl

import numpy as np
import scipy.sparse as sp


class DeepWalkDataset:

    def __init__(self, data, walk_length):

        datapath = 'data/' + data + '/' + data

        self.net, self.node2id, self.id2node, self.sm = ReadTxtNet(datapath+'-net.txt', True)

        self.graph = net2graph(self.sm)
        self.walk_lengh = walk_length


        # self.labels = self.read_label(datapath+'-label.txt')

    def sample_randomwalk(self, seeds):

        return dgl.contrib.sampling.random_walk(self.graph, seeds, 1, self.walk_lengh-1)



def ReadTxtNet(file_path="", undirected=True):
    """ Read the txt network file.
    Notations: The network is unweighted.
    Parameters
    ----------
    file_path str : path of network file
    undirected bool : whether the edges are undirected
    Return
    ------
    net dict : a dict recording the connections in the graph
    node2id dict : a dict mapping the nodes to their embedding indices
    id2node dict : a dict mapping nodes embedding indices to the nodes

    The reference for this ReadTxtNet is
        : github.com/ShawXh/DeepWalk-dgl/
    """
    node2id = {}
    id2node = {}
    cid = 0

    src = []
    dst = []
    net = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            n1, n2 = list(map(int, line.strip().split(" ")[:2]))
            if n1 not in node2id:
                node2id[n1] = cid
                id2node[cid] = n1
                cid += 1
            if n2 not in node2id:
                node2id[n2] = cid
                id2node[cid] = n2
                cid += 1

            n1 = node2id[n1]
            n2 = node2id[n2]
            if n1 not in net:
                net[n1] = {n2: 1}
                src.append(n1)
                dst.append(n2)
            elif n2 not in net[n1]:
                net[n1][n2] = 1
                src.append(n1)
                dst.append(n2)

            if undirected:
                if n2 not in net:
                    net[n2] = {n1: 1}
                    src.append(n2)
                    dst.append(n1)
                elif n1 not in net[n2]:
                    net[n2][n1] = 1
                    src.append(n2)
                    dst.append(n1)

    print("node num: %d" % len(net))
    print("edge num: %d" % len(src))
    assert max(net.keys()) == len(net) - 1, "error reading net, quit"

    sm = sp.coo_matrix(
        (np.ones(len(src)), (src, dst)),
        dtype=np.float32)

    return net, node2id, id2node, sm


def net2graph(net_sm):
    """ Transform the network to DGL graph
    Return
    ------
    G DGLGraph : graph by DGL
    """
    start = time.time()
    G = dgl.DGLGraph(net_sm)
    end = time.time()
    t = end - start
    print("Building DGLGraph in %.2fs" % t)
    return G