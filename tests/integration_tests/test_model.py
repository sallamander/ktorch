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

BATCH_SIZE = 4
HEIGHT = 28
WIDTH = 28
N_CHANNELS = 3
N_CLASSES = 3


def get_mock_callbacks(dirpath_model):
    """Mock callbacks to use during training

    The callback functionality tested via this fixture and the suite of tests
    in this file includes all of the callback functionality from the
    keras.callbacks module that is supported "out of the box" for the
    ktorch.model.Model class, as well as several ktorch.callbacks.

    Specific functionality tested includes some of the classes that are used by
    default in the ktorch.model.Model class (i.e. BaseLogger, CallbackList,
    History, and ProgbarLogger) as well as others that users can specify (i.e.
    CSVLogger, EarlyStopping, and ModelCheckpoint). This fixture simply returns
    instantiated versions of all of the latter, as well as a some of the
    ktorch.callbacks.

    :param dirpath_model: directory path where model weights / histories/ etc.
     will be stored
    :type dirpath_model: str
    :return: callbacks to use during training
    :rtype: list[object]
    """

    fpath_history = os.path.join(dirpath_model, 'history.csv')
    if os.path.exists(fpath_history):
        os.remove(fpath_history)
    csv_logger = CSVLogger(filename=os.path.join(dirpath_model, 'history.csv'))
    early_stopping = EarlyStopping(monitor='loss', patience=2, mode='min')

    fpath_weights = os.path.join(dirpath_model, 'weights.pt')
    if os.path.exists(fpath_weights):
        os.remove(fpath_weights)
    model_checkpoint = ModelCheckpoint(
        filepath=fpath_weights, monitor='loss', save_weights_only=True,
        mode='min'
    )

    if os.path.exists(os.path.join(dirpath_model, 'tensorboard')):
        shutil.rmtree(os.path.join(dirpath_model, 'tensorboard'))
    tensorboard = TensorBoard(
        log_dir=os.path.join(dirpath_model, 'tensorboard')
    )

    callbacks = [csv_logger, early_stopping, model_checkpoint, tensorboard]
    return callbacks


def get_mock_generator():
    """Mock generator object to use during training

    This generator simply returns randomly constructed torch tensors to use as
    mock inputs and targets for training.

    :return: inputs and targets to use for training
    :rtype: tuple(torch.Tensor)
    """

    while True:
        mock_inputs = torch.tensor(np.random.random(
            (BATCH_SIZE, N_CHANNELS, HEIGHT, WIDTH),
        ), dtype=torch.float)
        mock_targets = torch.tensor(np.random.randint(
            0, N_CLASSES, size=BATCH_SIZE
        ), dtype=torch.long)
        yield (mock_inputs, mock_targets)


def get_mock_metrics():
    """Mock metrics to use durining training

    :return: metrics to use durining training
    :rtype: list[object]
    """

    metrics = [categorical_accuracy, TopKCategoricalAccuracy(k=2)]
    return metrics


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


def check_fit_generator(dirpath_model, gpu_id=None):
    """Run fit_generator and assert that it ran as expected

    :param dirpath_model: directory path to save the model output to
    :type dirpath_model: str
    :param gpu_id: GPU to run the model on; if None, it will be run on the CPU
    :type gpu_id: int
    """

    mock_callbacks = get_mock_callbacks(dirpath_model)
    mock_generator = get_mock_generator()
    mock_metrics = get_mock_metrics()

    mock_network = ToyNetwork(
        input_shape=(N_CHANNELS, HEIGHT, WIDTH), n_classes=N_CLASSES
    )
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

    assert os.path.exists(os.path.join(dirpath_model, 'tensorboard'))
    assert os.path.exists(os.path.join(dirpath_model, 'history.csv'))
    assert os.path.exists(os.path.join(dirpath_model, 'weights.pt'))
    assert len(os.listdir(os.path.join(dirpath_model, 'tensorboard'))) == 1

    df_history = pd.read_csv(os.path.join(dirpath_model, 'history.csv'))
    assert set(df_history.columns) == {
        'epoch', 'loss', 'val_loss', 'categorical_accuracy',
        'val_categorical_accuracy', 'top_2_categorical_accuracy',
        'val_top_2_categorical_accuracy'
    }


class TestModel_CPU():
    """Test for Model when running on a CPU"""

    def test_fit_generator(self):
        """Test fit_generator

        This is more or less a smoke test, but does perform a couple of checks
        to ensure that the expected files are saved in the expected places, and
        that those files contained the expected contents (namely the
        'history.csv'). See `check_fit_generator` for details.
        """

        dirpath_model = os.path.join(tempdir, 'test_model_cpu')
        check_fit_generator(dirpath_model, gpu_id=None)


@pytest.mark.skip_gpu_tests
class TestModel_GPU():
    """Test for Model when running on a GPU"""

    def test_fit_generator(self):
        """Test fit_generator

        This is more or less a smoke test, but does perform a couple of checks
        to ensure that the expected files are saved in the expected places, and
        that those files contained the expected contents (namely the
        'history.csv'). See `check_fit_generator` for details.
        """

        dirpath_model = os.path.join(tempdir, 'test_model_gpu')
        check_fit_generator(dirpath_model, gpu_id=0)
