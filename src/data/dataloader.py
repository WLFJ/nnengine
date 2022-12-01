import numpy as np


class Dataset(object):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data()
        self.shape = self.data.shape

    def load_data(self):
        data = np.load(self.data_path)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=False, padding=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if padding:
            self.padding()
        self.index = 0
        self.length = len(dataset)
        self.batch_num = self.length // self.batch_size
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)

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
        padding_num = self.batch_size - len(self.dataset) % self.batch_size
        padding_data = np.zeros((padding_num, *self.dataset.shape[1:]))
        self.dataset = np.concatenate((self.dataset, padding_data), axis=0)


class DataSpliter:
    def __init__(self, dataset, ratio):
        self.dataset = dataset
        self.ratio = ratio
        self.length = len(dataset)
        self.split_num = int(self.length * self.ratio)

    def split(self):
        indexes = np.arange(self.length)
        np.random.shuffle(indexes)
        train_indexes = indexes[:self.split_num]
        test_indexes = indexes[self.split_num:]
        train_data = self.dataset[train_indexes]
        test_data = self.dataset[test_indexes]
        return train_data, test_data
