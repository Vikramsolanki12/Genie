import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import JetDataset
from graph_utils import create_graph
from models import ContrastiveModel


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def augment_graph(data):
    x = data.x.clone()
    noise = torch.randn_like(x) * 0.01
    data_aug = data.clone()
    data_aug.x = x + noise
    return data_aug


def contrastive_loss(z1, z2):
    return 1 - torch.cosine_similarity(z1, z2).mean()


dataset = JetDataset("data/quark-gluon_data-set_n139306.hdf5", max_samples=1000)

graphs = []
for i in range(len(dataset)):
    img, label = dataset[i]
    graphs.append(create_graph(img, label))

loader = DataLoader(graphs, batch_size=2, shuffle=True)
model = ContrastiveModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        aug1 = augment_graph(data)
        aug2 = augment_graph(data)
        z1 = model(aug1)
        z2 = model(aug2)
        loss = contrastive_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)