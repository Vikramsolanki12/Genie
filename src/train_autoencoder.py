import torch
from torch.utils.data import DataLoader

from dataset import JetDataset
from models import JetAutoencoder


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dataset = JetDataset("data/quark-gluon_data-set_n139306.hdf5", max_samples=20000)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = JetAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for images, _ in loader:
        images = images.to(device)
        recon = model(images)
        recon = recon[:, :, :125, :125]
        loss = criterion(recon, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)