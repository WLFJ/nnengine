import pickle
from tqdm import tqdm
import numpy as np
from lightGE.data import DataLoader

from lightGE.core.tensor import Tensor


class Trainer(object):

    def __init__(self, model, optimizer, loss_fun, config, schedule=None):
        self.m = model
        self.opt = optimizer
        self.sche = schedule
        self.lf = loss_fun

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.save_path = config['save_path']

    def train(self, train_dataset, eval_dataset):
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=self.shuffle)
        eval_dataloader = DataLoader(eval_dataset, self.batch_size, shuffle=self.shuffle)

        min_eval_loss, best_epoch = float('inf'), 0
        bar = tqdm(range(self.epochs))
        for epoch_idx in bar:
            train_loss = self._train_epoch(train_dataloader)
            eval_loss = self._eval_epoch(eval_dataloader)

            bar.set_description("Epoch: {}, ".format(epoch_idx) + 'training loss: {},'.format(train_loss) +
                                'validation loss: {}'.format(eval_loss))

            if self.sche is not None:
                self.sche.step(eval_loss)

            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                self.save_model(self.save_path)
                best_epoch = epoch_idx

        print("Best epoch: {}, Best validation loss: {}".format(best_epoch, min_eval_loss))

        return min_eval_loss

    def _train_epoch(self, train_dataloader) -> [float]:
        losses = []
        for batch in train_dataloader:
            y_truth = Tensor(batch.labels, autograd=False)
            y_pred = self.m(Tensor(batch.data, autograd=False))
            loss: Tensor = self.lf(y_pred, y_truth)
            loss.backward()
            self.opt.step(loss)
            losses.append(loss.data)
        return np.mean(losses)

    def _eval_epoch(self, eval_dataloader):
        losses = []
        for batch in eval_dataloader:
            y_truth = Tensor(batch.labels, autograd=False)
            y_pred = self.m(Tensor(batch.data, autograd=False))
            loss: Tensor = self.lf(y_pred, y_truth)
            losses.append(loss.data)
        return np.mean(losses)

    def load_model(self, cache_name):
        [self.m, self.opt, self.sche] = pickle.load(open(cache_name, 'rb'))

    def save_model(self, cache_name):
        pickle.dump([self.m, self.opt, self.sche], open(cache_name, 'wb'))
