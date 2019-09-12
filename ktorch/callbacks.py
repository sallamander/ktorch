"""Callbacks for training with PyTorch

Reference Implementations in:
    - https://github.com/keras-team/keras/blob/master/keras/callbacks.py

Keras callbacks that work out of the box with the ktorch.Model class and that
are exposed from this module by importing them:
    - CallbackList
    - CSVLogger
    - EarlyStopping
    - History
    - ModelCheckpoint
    - ProgbarLogger
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.callbacks import (
    Callback, CallbackList, CSVLogger, EarlyStopping, History, ModelCheckpoint,
    ProgbarLogger
)
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard.writer import FileWriter


class LRFinder(Callback):
    """Callback for finding the optimal learning rate range

    This callback adjusts the learning rate linearly from `min_lr` to `max_lr`
    during the `n_epochs` of training, recording the loss after each training
    step. At the end of training, it generates and saves plots of the loss
    (both smoothed and unsmoothed) as the learning rate changes.

    Original reference paper: https://arxiv.org/abs/1506.01186
    Reference implementation: https://docs.fast.ai/callbacks.lr_finder.html
    """

    def __init__(self, dirpath_results, n_steps_per_epoch,
                 min_lr=1e-5, max_lr=1e-2, n_epochs=1):
        """Init

        :param dirpath_results: directory path to store the results (history
         CSV and plots) in
        :type dirpath_results: str
        :param n_steps_per_epoch: number of training steps per epoch
        :type n_steps_per_epoch: int
        :param min_lr: minimum learning rate to use during training
        :type min_lr: float
        :param max_lr: maximum learning rate to use during training
        :type max_lr: float
        :param n_epochs: number of epochs to train for
        :type n_epochs: int
        """

        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        if not os.path.exists(dirpath_results):
            os.mkdir(dirpath_results)
        self.dirpath_results = dirpath_results

        self.n_total_iterations = n_steps_per_epoch * n_epochs
        self.averaging_window = int(0.05 * self.n_total_iterations)
        self.iteration = 0
        self.alpha = 0.98

        self.history = {}
        self.df_history = None
        # needs to be set with self.set_model
        self.model = None

    def _calc_learning_rate(self):
        """Calculate the learning rate at a given step

        :return: the new learning rate to use
        :rtype: float
        """

        pct_of_iterations_complete = self.iteration / self.n_total_iterations
        new_learning_rate = (
            self.min_lr + (self.max_lr - self.min_lr) *
            pct_of_iterations_complete
        )

        return new_learning_rate

    def _plot_loss(self, logscale=True):
        """Plot the unsmoothed loss throughout the course of training

        :param logscale: if True, plot using logscale for the x-axis
        :type logscale: bool
        """

        learning_rates = (
            self.history['lr'][10:-int(self.averaging_window * 0.5)]
        )
        loss_values = (
            self.history['loss'][10:-int(self.averaging_window * 0.5)]
        )

        _, ax = plt.subplots(1, 1)

        ax.plot(learning_rates, loss_values)

        min_loss = np.min(loss_values)
        early_loss_average = np.average(loss_values[10:self.averaging_window])
        # set the y-axis of the plot so that big jumps in the loss don't leave
        # the remainder of the plot un-interpretable; this won't really affect
        # the interpretation of the plot, since a spike off plot still tells us
        # that the learning rate is too high
        plt.ylim(
            min_loss * 0.90,
            max(loss_values[0] * 1.10, early_loss_average * 1.5)
        )
        if logscale:
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')

    def _plot_lr(self, logscale=True):
        """Plot the learning rate over the course of training

        :param logscale: if True, plot using logscale for the y-axis
        :type logscale: bool
        """

        plt.plot(self.df_history['iterations'], self.df_history['lr'])
        if logscale:
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')

    def _plot_smoothed_loss(self, logscale=True):
        """Plot the smoothed loss throughout training

        :param bool logscale: if True, plot using logscale for the x-axis
        """

        learning_rates = (
            self.history['lr'][10:-int(self.averaging_window * 0.5)]
        )
        smoothed_loss = (
            self.history['smoothed_loss'][10:-int(self.averaging_window * 0.5)]
        )

        _, ax = plt.subplots(1, 1)

        ax.plot(learning_rates, smoothed_loss)

        min_loss = np.min(smoothed_loss)
        early_loss_average = (
            np.average(smoothed_loss[10:self.averaging_window])
        )
        # set the y-axis of the plot so that big jumps in the loss don't leave
        # the remainder of the plot un-interpretable; this won't really affect
        # the interpretation of the plot, since a spike off plot still tells us
        # that the learning rate is too high
        plt.ylim(
            min_loss * 0.90,
            max(smoothed_loss[0] * 1.10, early_loss_average * 1.5)
        )
        if logscale:
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Smoothed Loss')

    def on_batch_end(self, _, logs=None):
        """Record previous batch statistics and update the learning rate

        :param int epochs: current epoch
        :param dict(list) logs: training logs as dictionary
        """

        logs = logs or {}
        self.iteration += 1

        current_learning_rates = [
            parameter_group['lr']
            for parameter_group in self.model.optimizer.param_groups
        ]
        assert len(set(current_learning_rates)) == 1

        current_learning_rate = current_learning_rates[0]
        self.history.setdefault('lr', []).append(current_learning_rate)
        self.history.setdefault('iterations', []).append(self.iteration)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        smoothed_loss = self.history.get('smoothed_loss', [0])[-1]
        smoothed_loss = (
            self.alpha * logs['loss'] +
            (1 - self.alpha) * smoothed_loss
        )
        smoothed_loss = smoothed_loss / (1 - self.alpha ** self.iteration)
        self.history.setdefault('smoothed_loss', []).append(smoothed_loss)

        min_loss = np.min(self.history['smoothed_loss'])
        window = int(0.10 * self.n_total_iterations)
        windowed_smoothed_loss = np.average(
            self.history['smoothed_loss'][-window:]
        )
        if windowed_smoothed_loss >= (5 * min_loss):
            self.model.stop_training = True

        new_learning_rate = self._calc_learning_rate()
        for parameter_group in self.model.optimizer.param_groups:
            parameter_group['lr'] = new_learning_rate

    def on_train_begin(self, logs=None):
        """Initialize the learning rate

        :param dict(list) logs: training logs as dictionary
        """

        learning_rates = [
            parameter_group['lr']
            for parameter_group in self.model.optimizer.param_groups
        ]
        n_different_learning_rates = len(set(learning_rates))
        if n_different_learning_rates != 1:
            msg = (
                'LRFinder is not compatible with using multiple learning '
                'rates. There are {} unique learning rates being used: {}.'
            ).format(n_different_learning_rates, str(set(learning_rates)))
            raise ValueError(msg)

        logs = logs or {}
        for parameter_group in self.model.optimizer.param_groups:
            parameter_group['lr'] = self.min_lr

    def on_train_end(self, logs=None):
        """Create self.df_history

        :param dict(list) logs: training logs as dictionary
        """

        self.df_history = pd.DataFrame(self.history)

        fpath_df_history = os.path.join(self.dirpath_results, 'history.csv')
        self.df_history.to_csv(fpath_df_history, index=False)

        for logscale in [True, False]:
            self._plot_loss(logscale=logscale)

            fpath_loss = os.path.join(self.dirpath_results, 'unsmoothed_loss')
            if logscale:
                fpath_loss += '_logscale.png'
            else:
                fpath_loss += '.png'
            plt.savefig(fpath_loss)
            plt.clf()

            self._plot_lr(logscale=logscale)

            fpath_lr = os.path.join(self.dirpath_results, 'lr')
            if logscale:
                fpath_lr += '_logscale.png'
            else:
                fpath_lr = '.png'
            plt.savefig(fpath_lr)
            plt.clf()

            self._plot_smoothed_loss(logscale=logscale)

            fpath_loss_smoothed = os.path.join(
                self.dirpath_results, 'smoothed_loss'
            )
            if logscale:
                fpath_loss_smoothed += '_logscale.png'
            else:
                fpath_loss_smoothed += '.png'
            plt.savefig(fpath_loss_smoothed)
            plt.clf()


class TensorBoard(Callback):
    """Tensorboard basic visualizations

    Reference Implementation in:
        - https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    """

    def __init__(self, log_dir='./logs', update_freq='batch'):
        """Init

        :param log_dir: the directory path to save the log files for
         TensorBoard to parse
        :type log_dir: str
        :param update_freq: how often to write loss and metrics, one of 'epoch'
         or 'batch'
        :type update_freq: str
        """

        super().__init__()

        self.update_freq = update_freq
        self.writer = FileWriter(log_dir=log_dir)
        self.samples_seen = 0

    def _write_logs(self, logs, index):
        """Write log values to the log files

        :param logs: holds the loss and metric values computed at the most
         recent interval (batch or epoch)
        :type logs: dict
        :param index: if update_freq='batch', the total number of samples that
         have been seen, else if update_freq='epoch', the epoch index
        :type index: int
        """

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = Summary(
                value=[Summary.Value(tag=name, simple_value=value)]
            )
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        """Save loss and metric statistics for the most recent batch

        :param batch: index of the most recent batch
        :type batch: int
        :param logs: holds the loss and metric values computed on the most
         recent batch
        :type logs: dict
        """

        if self.update_freq == 'batch':
            self.samples_seen += logs['size']
            self._write_logs(logs, self.samples_seen)

    def on_epoch_end(self, epoch, logs=None):
        """Save loss and metric statistics for the most recent epoch

        :param epoch: index of the most recent epoch
        :type epoch: int
        :param logs: holds the loss and metric values computed on the most
         recent epoch
        :type logs: dict
        """

        if self.update_freq == 'epoch':
            self._write_logs(logs, epoch)
