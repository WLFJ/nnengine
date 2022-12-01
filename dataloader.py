import numpy as np


class AbastractDataset(object):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError


class MnistDataset(AbastractDataset):
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


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=False, padding=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        self.length = len(dataset)
        self.batch_num = self.length // self.batch_size
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        if padding:
            self.padding()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.batch_num:
            batch = []
            for i in range(self.batch_size):
                idx = self.indexes[self.index * self.batch_size + i]
                batch.append(self.dataset[idx])
            self.index += 1
            return batch
        else:
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.indexes)
            raise StopIteration

    def __len__(self):
        return self.batch_num

    def padding(self):
