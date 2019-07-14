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


from keras.callbacks import (
    Callback, CallbackList, CSVLogger, EarlyStopping, History, ModelCheckpoint,
    ProgbarLogger
)
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard.writer import FileWriter


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
        self.writer = FileWriter(logdir=log_dir)
        self.samples_seen = 0

    def _write_logs(self, logs, index):
        """Write log values to the log files

        :param logs:
        :type logs:
        :param index:
        :type index:
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
