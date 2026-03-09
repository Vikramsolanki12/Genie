import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class JetDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        f = h5py.File(file_path, "r")
        images = f["X_jets"][:]
        labels = f["y"][:]
        f.close()

        if max_samples:
            images = images[:max_samples]
            labels = labels[:max_samples]

        # convert to PyTorch format
        images = np.transpose(images, (0, 3, 1, 2))
        images = images / images.max()
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(int), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]