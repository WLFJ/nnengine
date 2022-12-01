from src.data.dataloader import Dataset
import numpy as np


class MnistDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(MnistDataset, self).__init__(data_path, transform)

    def load_data(self):
        data = np.load(self.data_path)
        return data

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img
