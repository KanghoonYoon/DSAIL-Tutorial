import torch as th

from dgl import DGLGraph
import dgl


from argparse import ArgumentParser




if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data", type=str, default='cora')


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

    model = GraphCon







    print("END")

