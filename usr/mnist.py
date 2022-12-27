from lightGE.data.dataloader import Dataset
from lightGE.core import Tensor, Model, Conv2d, Linear, ReLu, Dropout, MaxPool2d, Sequential
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


class MNIST(Model):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = Sequential(
            [Conv2d(1, 10, filter_size=5),
             MaxPool2d(filter_size=2),
             ReLu()])

        self.conv2 = Sequential([
            Conv2d(10, 20, filter_size=5),
            Dropout(),
            MaxPool2d(filter_size=2),
            ReLu()])
        self.fc1 = Sequential([Linear(320, 50), ReLu()])
        self.fc2 = Sequential([Dropout(), Linear(50, 10)])

        self.parameters = []
        self.parameters += self.conv1.get_parameters()
        self.parameters += self.conv2.get_parameters()
        self.parameters += self.fc1.get_parameters()
        self.parameters += self.fc2.get_parameters()

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.softmax()

    def get_parameters(self):
        return self.parameters
