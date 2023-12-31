import tqdm

from lightGE.data.dataloader import Dataset
from lightGE.core import Tensor, Model, Conv2d, Linear, ReLu, Dropout2d, MaxPool2d, Sequential
from lightGE.utils import SGD, Trainer, nll_loss
import numpy as np
import gzip
from lightGE.data import DataLoader


class MnistDataset(Dataset):
    def __init__(self):
        super(MnistDataset, self).__init__()

    def load_data(self, data_dir):
        def extract_data(filename, num_data, head_size, data_size):
            with gzip.open(filename) as bytestream:
                bytestream.read(head_size)
                buf = bytestream.read(data_size * num_data)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
            return data

        data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
        trX = data.reshape((60000, 28, 28, 1))

        data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
        trY = data.reshape((60000))

        data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
        teX = data.reshape((10000, 28, 28, 1))

        data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
        teY = data.reshape((10000))

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        x = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int32)

        data_index = np.arange(x.shape[0])
        np.random.shuffle(data_index)
        # data_index = data_index[:128]
        x = x[data_index, :, :, :]
        y = y[data_index]
        y_vec = np.zeros((len(y), 10), dtype=np.float64)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        x /= 255.
        x = x.transpose(0, 3, 1, 2)

        self.x = x
        self.y = y_vec


class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv1 = Sequential(
            [Conv2d(1, 10, filter_size=5),
             MaxPool2d(filter_size=2),
             ReLu()])

        self.conv2 = Sequential([
            Conv2d(10, 20, filter_size=5),
            Dropout2d(),
            MaxPool2d(filter_size=2),
            ReLu()])
        self.fc1 = Sequential([Linear(320, 50), ReLu()])
        self.fc2 = Sequential([Dropout2d(), Linear(50, 10)])

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x.softmax().log()


def evaluate(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = 0
    bar = tqdm.tqdm(dataloader)
    i = 0
    for x, y in bar:
        y_pred = model(Tensor(x))
        y_pred = np.argmax(y_pred.data, axis=1)
        y = np.argmax(y.data, axis=1)
        correct += np.sum(y_pred == y)
        i += 128
        bar.set_description("acc: {}".format(correct / i))
    return correct / len(dataset)


if __name__ == '__main__':
    mnist_dataset = MnistDataset()
    mnist_dataset.load_data('./res/mnist/')
    train_dataset, eval_dataset = mnist_dataset.split(0.7)

    m = MNIST()
    opt = SGD(parameters=m.parameters(), lr=0.01)

    cache_path = 'tmp/mnist.pkl'
    trainer = Trainer(model=m, optimizer=opt, loss_fun=nll_loss,
                      config={'batch_size': 128,
                              'epochs': 10,
                              'shuffle': False,
                              'save_path': cache_path})

    # trainer.train(train_dataset, eval_dataset)
    trainer.load_model('./tmp/mnist_bst.pkl')
    # trainer.train(train_dataset, eval_dataset)
    evaluate(trainer.m, eval_dataset)
