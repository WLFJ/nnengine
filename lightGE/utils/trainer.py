import pickle
from tqdm import tqdm
import numpy as np
from lightGE.data import DataLoader

from lightGE.core.tensor import Tensor

import logging

logging.basicConfig(level=logging.INFO)


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
        for epoch_idx in range(self.epochs):
            train_loss = self._train_epoch(train_dataloader)
            eval_loss = self._eval_epoch(eval_dataloader)

            if self.sche is not None:
                self.sche.step(eval_loss)

            logging.info("Lr: {}".format(self.opt.lr))

            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                self.save_model(self.save_path)
                best_epoch = epoch_idx

        logging.info("Best epoch: {}, Best validation loss: {}".format(best_epoch, min_eval_loss))

        return min_eval_loss

    def _train_epoch(self, train_dataloader) -> [float]:
        self.m.train()
        losses = []

        bar = tqdm(train_dataloader)
        batch_idx = 0

        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = Tensor(batch_y, autograd=False)
            y_pred = self.m(Tensor(batch_x, autograd=False))
            loss: Tensor = self.lf(y_pred, y_truth)
            loss.backward()
            self.opt.step()
            losses.append(loss.data)

            bar.set_description("Batch: {}/{} ".format(batch_idx, len(train_dataloader)) +
                                'Training loss: {},'.format(np.mean(losses)))

        return np.mean(losses)

    def _eval_epoch(self, eval_dataloader):
        self.m.eval()
        losses = []
        for batch_x, batch_y in eval_dataloader:
            y_truth = Tensor(batch_y, autograd=False)
            y_pred = self.m(Tensor(batch_x, autograd=False))
            loss: Tensor = self.lf(y_pred, y_truth)
            losses.append(loss.data)
        logging.info("Validation loss: {}".format(np.mean(losses)), )

        return np.mean(losses)

    def load_model(self, cache_name):
        [self.m, self.opt, self.sche] = pickle.load(open(cache_name, 'rb'))

    def save_model(self, cache_name):
        pickle.dump([self.m, self.opt, self.sche], open(cache_name, 'wb'))
