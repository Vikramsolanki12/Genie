import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def image_to_pointcloud(image, max_nodes=150):
    nodes = []
    for c in range(3):
        for i in range(125):
            for j in range(125):
                val = image[c, i, j]
                if val > 0:
                    nodes.append([i / 125, j / 125, val, c])

    nodes = np.array(nodes)

    if len(nodes) > max_nodes:
        idx = np.argsort(nodes[:, 2])[-max_nodes:]
        nodes = nodes[idx]

    return nodes


def build_edges(nodes, k=5):
    coords = nodes[:, :2]
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    edges = []
    for i in range(len(nodes)):
        for j in indices[i]:
            edges.append([i, j])

    edge_index = torch.tensor(edges).t().contiguous()
    return edge_index


def create_graph(image, label):
    nodes = image_to_pointcloud(image.numpy())
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = build_edges(nodes)
    y = torch.tensor([int(label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)