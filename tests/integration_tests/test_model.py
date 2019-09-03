"""Integration tests for model.py"""

import os
import shutil
from tempfile import tempdir

import numpy as np
import pandas as pd
import pytest
import torch
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from torch.nn import Conv2d, CrossEntropyLoss, Linear, Module, ReLU

from ktorch.callbacks import TensorBoard
from ktorch.metrics import categorical_accuracy, TopKCategoricalAccuracy
from ktorch.model import Model


class ToyNetwork(Module):
    """Small simple network for testing purposes"""

    def __init__(self, input_shape, n_classes):
        """Init

        :param input_shape: (n_channels, height, width) shape of a single input
         that will be passed to the forward method
        :type input_shape: tuple
        :param n_classes: number of classes in the final output layer
        :type n_classes: int
        """

        super().__init__()

        self.conv = Conv2d(
            in_channels=input_shape[0], out_channels=8,
            kernel_size=(1, 1), stride=(1, 1), bias=False
        )

        in_features = input_shape[1] * input_shape[2] * 8
        self.linear = Linear(in_features=in_features, out_features=n_classes)

    def forward(self, inputs):
        """Return the output of a forward pass of the ToyNetwork

        :param inputs: batch of input images, of shape
         (batch_size, n_channels, height, width)
        :type inputs: torch.Tensor
        :return: outputs of a ToyNetwork model, of shape
         (batch_size, n_classes)
        :rtype: torch.Tensor
        """

        layer = self.conv(inputs)
        layer = ReLU()(layer)

        layer = layer.view(layer.size(0), -1)
        outputs = self.linear(layer)

        return outputs


class TestModel():
    """Test for Model"""

    BATCH_SIZE = 4
    HEIGHT = 28
    WIDTH = 28
    N_CHANNELS = 3
    N_CLASSES = 3

    @pytest.fixture(scope='class')
    def mock_callbacks(self):
        """Mock callbacks to use during training

        The callback functionality tested in this suite includes all of the
        callback functionality from the keras.callbacks module that is
        supported "out of the box" for the ktorch.model.Model class, as well as
        several ktorch.callbacks.

        Functionality tested includes some of the classes that are used by
        default in the ktorch.model.Model class (i.e. BaseLogger, CallbackList,
        History, and ProgbarLogger) as well as others that users can specify
        (i.e.  CSVLogger, EarlyStopping, and ModelCheckpoint). This fixture
        simply returns instantiated versions of all of the latter, as well as a
        some of the ktorch.callbacks.

        :return: callbacks to use during training
        :rtype: list[object]
        """

        fpath_history = os.path.join(tempdir, 'history.csv')
        if os.path.exists(fpath_history):
            os.remove(fpath_history)
        csv_logger = CSVLogger(filename=os.path.join(tempdir, 'history.csv'))
        early_stopping = EarlyStopping(monitor='loss', patience=2, mode='min')

        fpath_weights = os.path.join(tempdir, 'weights.pt')
        if os.path.exists(fpath_weights):
            os.remove(fpath_weights)
        model_checkpoint = ModelCheckpoint(
            filepath=fpath_weights, monitor='loss', save_weights_only=True,
            mode='min'
        )

        if os.path.exists('/tmp/tensorboard'):
            shutil.rmtree('/tmp/tensorboard')
        tensorboard = TensorBoard(log_dir=os.path.join(tempdir, 'tensorboard'))

        callbacks = [csv_logger, early_stopping, model_checkpoint, tensorboard]
        return callbacks

    @pytest.fixture(scope='class')
    def mock_generator(self):
        """Mock generator object to use during training

        This generator simply returns randomly constructed np.ndarray objects
        to use as mock inputs and targets for training.

        :return: function object that can be instantiated to produce a
         generator
        :rtype: function
        """

        def generator():
            """Generator

            :return: inputs and targets to use for training
            :rtype: tuple(np.ndarray)
            """

            while True:
                mock_inputs = torch.tensor(np.random.random(
                    (self.BATCH_SIZE, self.N_CHANNELS,
                     self.HEIGHT, self.WIDTH),
                ), dtype=torch.float)
                mock_targets = torch.tensor(np.random.randint(
                    0, self.N_CLASSES, size=self.BATCH_SIZE
                ), dtype=torch.long)
                yield (mock_inputs, mock_targets)

        return generator

    @pytest.fixture
    def mock_metrics(self):
        """Mock metrics to use durining training

        :return: metrics to use durining training
        :rtype: list[object]
        """

        metrics = [categorical_accuracy, TopKCategoricalAccuracy(k=2)]
        return metrics

    def test_fit_generator__cpu(self, mock_callbacks, mock_generator,
                                mock_metrics):
        """Test fit_generator when running on a CPU

        This is more or less a smoke test, but does perform a couple of checks
        to ensure that the expected files are saved in the expected places, and
        that those files contained the expected contents (namely the
        'history.csv').

        :param mock_callbacks: mock_callbacks object fixture
        :type mock_callbacks: list[object]
        :param mock_generator: mock_generator object fixture
        :type mock_generator: function object
        :param mock_metrics: metrics to use during training
        :type mock_metrics: list[object]
        """

        mock_network = ToyNetwork(
            input_shape=(self.N_CHANNELS, self.HEIGHT, self.WIDTH),
            n_classes=self.N_CLASSES
        )
        mock_generator = mock_generator()
        mock_optimizer = torch.optim.Adam(
            params=mock_network.parameters(), lr=1e-4
        )

        model = Model(mock_network)
        model.compile(
            optimizer=mock_optimizer, loss=CrossEntropyLoss(),
            metrics=mock_metrics
        )
        model.fit_generator(
            generator=mock_generator, n_steps_per_epoch=10, n_epochs=5,
            validation_data=mock_generator, n_validation_steps=5,
            callbacks=mock_callbacks
        )

        assert os.path.exists('/tmp/tensorboard')
        assert os.path.exists('/tmp/history.csv')
        assert os.path.exists('/tmp/weights.pt')
        assert len(os.listdir('/tmp/tensorboard')) == 1

        df_history = pd.read_csv('/tmp/history.csv')
        assert set(df_history.columns) == {
            'epoch', 'loss', 'val_loss', 'categorical_accuracy',
            'val_categorical_accuracy', 'top_2_categorical_accuracy',
            'val_top_2_categorical_accuracy'
        }
