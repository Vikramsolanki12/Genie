import torch
from torch_geometric.loader import DataLoader

from dataset import JetDataset
from graph_utils import create_graph
from models import JetGNN


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dataset = JetDataset("data/quark-gluon_data-set_n139306.hdf5", max_samples=2000)

graphs = []
for i in range(len(dataset)):
    img, label = dataset[i]
    graphs.append(create_graph(img, label))

loader = DataLoader(graphs, batch_size=4, shuffle=True)
model = JetGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)