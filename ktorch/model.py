"""Class for training / evaluating pytorch networks

Reference Implementations:
- https://github.com/keras-team/keras/blob/master/keras/engine/training.py
"""

import numpy as np

from keras.callbacks import (
    BaseLogger, CallbackList, History, ProgbarLogger
)
from keras.engine.training_utils import make_batches
import torch


class Model(object):
    """Model for training / evaluating pytorch networks

    Reference Implementation:
    - https://github.com/keras-team/keras/blob/master/keras/engine/training.py
    """

    def __init__(self, network, device=None):
        """Init

        :param network: pytorch network to train or evaluate
        :type network: torch.nn.Module
        :param device: device to train the network on, e.g. 'cuda:0'
        :type device: str
        """

        self.network = network
        self.device = device

        self._compiled = False
        # these are set in the `compile` method
        self.optimizer = None
        self.loss = None
        self.metric_names = []
        self.metric_fns = []

        self.history = History()
        self.stop_training = False

    def _assert_compiled(self):
        """Raise a value error if the model is not compiled

        This is a convenience wrapper to avoid duplicating these lines in
        multiple methods.

        :raises: RuntimeError if `self._compiled` is not True
        """

        if not self._compiled:
            msg = ('Model must be compiled before training; please call '
                   'the `compile` method before training.')
            raise RuntimeError(msg)

    def _default_callbacks(self):
        """Return default callbacks automatically applied during training

        By default, the following callbacks are automatically applied during
        training:
        - tensorflow.keras.callbacks.BaseLogger
        - tensorflow.keras.callbacks.ProgbarLogger
        - tensorflow.keras.callbacks.History (which is the `Model.history`
          attribute set in `Model.__init__`)

        :return: callbacks automatically applied to every Model
        :rtype: list
        """

        default_callbacks = [BaseLogger(), self.history]
        return default_callbacks

    def compile(self, optimizer, loss, metrics=None):
        """Setup the model for training

        This sets `self.optimizer` and `self.loss` in place.

        :param optimizer: class name of the optimizer to use when training, one
         of those from `torch.optim` (e.g. `Adam`)
        :type optimizer: str
        :param loss: class name of the loss to use when training, one of those
         from `torch.nn` (e.g. `CrossEntropyLoss`)
        :type loss: str
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :raises AttributeError: if an invalid optimizer or loss function is
         specified
        """

        try:
            Optimizer = getattr(torch.optim, optimizer)
        except AttributeError:
            msg = (
                '`optimizer` must be a `str` representing an optimizer from '
                'the `torch.optim` package, and {} is not a valid one.'
            )
            raise AttributeError(msg.format(optimizer))
        self.optimizer = Optimizer(self.network.parameters())

        try:
            Loss = getattr(torch.nn, loss)
        except AttributeError:
            msg = (
                '`loss` must be a `str` representing a loss from '
                'the `torch.nn` package, and {} is not a valid one.'
            )
            raise AttributeError(msg.format(loss))
        self.loss = Loss()

        metrics = [] if not metrics else metrics
        for metric in metrics:
            metric_name = (
                metric.name if hasattr(metric, 'name') else metric.__name__
            )
            self.metric_names.append(metric_name)
            self.metric_fns.append(metric)

        self._compiled = True

    def evaluate(self, x, y, batch_size):
        """Evaluate the given data in test mode

        :param x: input data to use for evaluation
        :type x: torch.Tensor
        :param y: target data to use for evaluation
        :type y: torch.Tensor
        :param batch_size: number of samples to use per evaluation step
        :type batch_size: int
        :return: average metric values calculated between the outputs of the
         forward pass run on x and y
        :rtype: tuple(float)
        """

        self._assert_compiled()

        if self.device:
            self.network.to(self.device)

        batches = make_batches(x.shape[0], batch_size)
        metric_values_per_batch = []
        batch_sizes = []
        for idx_start, idx_end in batches:
            inputs = x[idx_start:idx_end]
            targets = y[idx_start:idx_end]

            n_obs = inputs.shape[0]
            batch_sizes.append(n_obs)

            test_outputs = self.test_on_batch(inputs, targets)
            metric_values_per_batch.append(test_outputs)

        validation_outputs = []
        for idx_value in range(len(test_outputs)):
            validation_outputs.append(
                np.average([
                    metric_values[idx_value]
                    for metric_values in metric_values_per_batch
                ], weights=batch_sizes)
            )
        return validation_outputs

    def evaluate_generator(self, generator, n_steps):
        """Evaluate the network on batches of data generated from `generator`

        :param generator: a generator yielding batches indefinitely, where each
         batch is a tuple of (inputs, targets)
        :type generator: generator
        :param n_steps: number of batches to evaluate on
        :type n_steps: int
        :return: average metric values calculated between the outputs of the
         forward pass run on the inputs from the generator and the targets
         produced from the generator
        :rtype: tuple(float)
        """

        self._assert_compiled()

        if self.device:
            self.network.to(self.device)

        metric_values_per_batch = []
        batch_sizes = []
        for _ in range(n_steps):
            inputs, targets = next(generator)
            n_obs = inputs.shape[0]
            batch_sizes.append(n_obs)

            test_outputs = self.test_on_batch(inputs, targets)
            metric_values_per_batch.append(test_outputs)

        validation_outputs = []
        for idx_value in range(len(test_outputs)):
            validation_outputs.append(
                np.average([
                    metric_values[idx_value]
                    for metric_values in metric_values_per_batch
                ], weights=batch_sizes)
            )
        return validation_outputs

    def fit(self, x, y, batch_size, n_epochs=1, callbacks=None,
            validation_data=None):
        """Trains the network on the given data for a fixed number of epochs

        :param x: input data to train on
        :type x: torch.Tensor
        :param y: target data to train on
        :type y: torch.Tensor
        :param batch_size: number of samples to use per forward and backward
         pass
        :type batch_size: int
        :param n_epochs: number of epochs (iterations of the dataset) to train
         the model
        :type n_epochs: int
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        :param validation_data: data on which to evaluate the loss and metrics
         at the end of each epoch
        :type validation_data: tuple(numpy.ndarray)
        """

        default_callbacks = self._default_callbacks()
        default_callbacks.append(ProgbarLogger(count_mode='samples'))
        if callbacks:
            default_callbacks.extend(callbacks)
        callbacks = CallbackList(default_callbacks)

        self._assert_compiled()

        if self.device:
            self.network.to(self.device)

        metrics = ['loss']
        if validation_data is not None:
            metrics.append('val_loss')
        for metric_name in self.metric_names:
            metrics.append(metric_name)
            if validation_data is not None:
                metrics.append('val_{}'.format(metric_name))

        index_array = np.arange(x.shape[0])

        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': n_epochs,
            'metrics': metrics,
            'steps': None,
            'samples': x.shape[0],
            'verbose': True
        })
        callbacks.set_model(self)

        callbacks.on_train_begin()
        for idx_epoch in range(n_epochs):
            if self.stop_training:
                break

            epoch_logs = {}
            callbacks.on_epoch_begin(idx_epoch)

            np.random.shuffle(index_array)
            batches = make_batches(len(index_array), batch_size)
            for idx_batch, (idx_start, idx_end) in enumerate(batches):
                batch_logs = {'batch': idx_batch, 'size': idx_end - idx_start}
                callbacks.on_batch_begin(idx_batch, batch_logs)

                inputs = x[index_array[idx_start:idx_end]]
                targets = y[index_array[idx_start:idx_end]]
                train_outputs = self.train_on_batch(inputs, targets)

                batch_logs['loss'] = train_outputs[0]
                it = zip(self.metric_names, train_outputs[1:])
                for metric_name, train_output in it:
                    batch_logs[metric_name] = train_output
                callbacks.on_batch_end(idx_batch, batch_logs)

                if self.stop_training:
                    break

            if validation_data:
                val_outputs = self.evaluate(
                    validation_data[0], validation_data[1], batch_size
                )

                epoch_logs['val_loss'] = val_outputs[0]
                it = zip(self.metric_names, val_outputs[1:])
                for metric_name, val_output in it:
                    metric_name = 'val_{}'.format(metric_name)
                    epoch_logs[metric_name] = val_output
            callbacks.on_epoch_end(idx_epoch, epoch_logs)
        callbacks.on_train_end()

    def fit_generator(self, generator, n_steps_per_epoch, n_epochs=1,
                      validation_data=None, n_validation_steps=None,
                      callbacks=None):
        """Train the network on batches of data generated from `generator`

        :param generator: a generator yielding batches indefinitely, where each
         batch is a tuple of (inputs, targets)
        :type generator: generator
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param n_epochs: number of epochs to train the model
        :type n_epochs: int
        :param validation_data: generator yielding batches to evaluate the loss
         on at the end of each epoch, where each batch is a tuple of (inputs,
         targets)
        :type validation_data: generator
        :param n_validation_steps: number of batches to evaluate on from
         `validation_data`
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        :raises RuntimeError: if only one of `validation_data` and
         `n_validation_steps` are passed in
        """

        default_callbacks = self._default_callbacks()
        default_callbacks.append(ProgbarLogger(count_mode='steps'))
        if callbacks:
            default_callbacks.extend(callbacks)
        callbacks = CallbackList(default_callbacks)

        self._assert_compiled()

        invalid_inputs = (
            (validation_data is not None and n_validation_steps is None) or
            (n_validation_steps is not None and validation_data is None)
        )
        if invalid_inputs:
            msg = (
                '`validation_data` and `n_validation_steps` must both be '
                'passed, or neither.'
            )
            raise RuntimeError(msg)

        if self.device:
            self.network.to(self.device)

        metrics = ['loss']
        if validation_data is not None:
            metrics.append('val_loss')
        for metric_name in self.metric_names:
            metrics.append(metric_name)
            if validation_data is not None:
                metrics.append('val_{}'.format(metric_name))

        callbacks.set_params({
            'epochs': n_epochs,
            'metrics': metrics,
            'steps': n_steps_per_epoch,
            'verbose': True
        })
        callbacks.set_model(self)

        callbacks.on_train_begin()
        for idx_epoch in range(n_epochs):
            if self.stop_training:
                break

            epoch_logs = {}
            callbacks.on_epoch_begin(idx_epoch)

            for idx_batch in range(n_steps_per_epoch):
                batch_logs = {'batch': idx_batch, 'size': 1}
                callbacks.on_batch_begin(idx_batch, batch_logs)

                inputs, targets = next(generator)
                train_outputs = self.train_on_batch(inputs, targets)

                batch_logs['loss'] = train_outputs[0]
                it = zip(self.metric_names, train_outputs[1:])
                for metric_name, train_output in it:
                    batch_logs[metric_name] = train_output
                callbacks.on_batch_end(idx_batch, batch_logs)

                if self.stop_training:
                    break

            if validation_data:
                val_outputs = self.evaluate_generator(
                    validation_data, n_validation_steps
                )

                epoch_logs['val_loss'] = val_outputs[0]
                it = zip(self.metric_names, val_outputs[1:])
                for metric_name, val_output in it:
                    metric_name = 'val_{}'.format(metric_name)
                    epoch_logs[metric_name] = val_output
            callbacks.on_epoch_end(idx_epoch, epoch_logs)
        callbacks.on_train_end()

    def load_weights(self, fpath_weights):
        """Loads all layer weights from the provided `fpath_weights`

        :param fpath_weights: fpath_weights to load the model from
        :type fpath_weights: str
        """

        self.network.load_state_dict(torch.load(fpath_weights))

    def predict(self, x, batch_size):
        """Generate output predictions for the input samples

        :param x: input data to predict on
        :type x: torch.Tensor
        :param batch_size: number of samples to predict on at one time
        :type batch_size: int
        :return: array of predictions
        :rtype: numpy.ndarray
        """

        batches = make_batches(len(x), batch_size)
        predictions_per_batch = []
        for idx_batch, (idx_start, idx_end) in enumerate(batches):
            inputs = x[idx_start:idx_end]
            predictions = self.network.forward(inputs)
            predictions_per_batch.append(predictions)

        batch_predictions = torch.cat(predictions_per_batch)
        return batch_predictions

    def save_weights(self, fpath_weights, overwrite=True):
        """Dumps all layers and weights to the provided `fpath_weights`

        The weights can be loaded into a `Model` with the same topology using
        the `Model.load_weights` method.

        :param fpath_weights: fpath_weights to save the model to
        :type fpath_weights: str
        :param overwrite: overwrite an existing file at `fpath_weights`
         (if present); only True is currently supported
        :type overwrite: bool
        """

        assert overwrite, '`overwrite=False` is not supported!'
        torch.save(self.network.state_dict(), fpath_weights)

    def test_on_batch(self, inputs, targets):
        """Evaluate the model on a single batch of samples

        :param inputs: inputs to predict on
        :type inputs: torch.Tensor
        :param targets: targets to compare model predictions to
        :type targets: torch.Tensor
        :return: metrics calculated between the outputs of the forward pass and
         the targets
        :rtype: tuple(float)
        """

        self._assert_compiled()

        self.network.train(mode=False)
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self.network(inputs)
        loss = self.loss(outputs, targets)

        test_outputs = [loss.tolist()]
        for metric_fn in self.metric_fns:
            test_outputs.append(metric_fn(targets, outputs))

        return test_outputs

    def train_on_batch(self, inputs, targets):
        """Run a single forward / backward pass on a single batch of data

        :param inputs: inputs to use in the forward / backward pass
        :type inputs: torch.Tensor
        :param targets: targets to use in the forward / backward pass
        :type targets: torch.Tensor
        :return: metrics calculated between the outputs of the forward pass and
         the targets
        :rtype: tuple(float)
        """

        self._assert_compiled()

        self.network.train(mode=True)
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self.network(inputs)
        loss = self.loss(outputs, targets)

        train_outputs = [loss.tolist()]
        for metric_fn in self.metric_fns:
            train_outputs.append(metric_fn(targets, outputs))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_outputs
