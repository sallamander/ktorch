"""Unit tests for model"""

from itertools import product
from unittest.mock import create_autospec, patch, MagicMock

import numpy as np
from keras.callbacks import (
    BaseLogger, CallbackList, History, ProgbarLogger
)
import pytest
import torch

from ktorch.model import Model


class TestModel(object):
    """Tests for Model"""

    def test_init(self):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.
        """

        network = MagicMock()
        device = MagicMock()

        model = Model(network, device)
        assert id(network) == id(model.network)
        assert id(device) == id(model.device)
        assert not model._compiled
        assert not model.optimizer
        assert not model.loss
        assert model.history
        assert isinstance(model.history, History)
        assert not model.stop_training
        assert not model.metric_names
        assert not model.metric_fns

        model = Model(network)
        assert not model.device

    def test_assert_compiled(self):
        """Test _assert_compiled method"""

        model = MagicMock()
        model._assert_compiled = Model._assert_compiled

        model._compiled = False
        with pytest.raises(RuntimeError):
            model._assert_compiled(self=model)

        model._compiled = True
        model._assert_compiled(self=model)

    def test_default_callbacks(self):
        """Test _default_callbacks method"""

        model = MagicMock()
        model.history = MagicMock()
        model._default_callbacks = Model._default_callbacks

        callbacks = model._default_callbacks(self=model)
        assert isinstance(callbacks[0], BaseLogger)
        assert id(callbacks[1]) == id(model.history)

    def test_compile(self):
        """Test compile method

        This tests several things:
        - An `AttributeError` is raised if an invalid optimizer or loss
          function is passed in
        - `Model.loss` and `Model.optimizer` are set correctly when a valid
          optimizer and loss are passed in
        - `Model._compiled` is True after `compile` is called
        """

        model = MagicMock()
        model.optimizer = None
        model.loss = None
        model._compiled = False
        model.compile = Model.compile
        model.metric_names = []
        model.metric_fns = []

        network = MagicMock()
        parameters_fn = MagicMock()
        parameters_fn.return_value = [
            torch.nn.Parameter(torch.randn((64, 64)))
        ]
        network.parameters = parameters_fn
        model.network = network

        valid_optimizers = ['Adam', 'RMSprop']
        valid_losses = ['BCELoss', 'CrossEntropyLoss', 'L1Loss']

        mock_metric = MagicMock()
        mock_metric.name = 'mock_metric'
        metrics = [mock_metric]

        for optimizer, loss in product(valid_optimizers, valid_losses):
            assert not model.optimizer
            assert not model.loss
            assert not model._compiled

            model.compile(
                self=model, optimizer=optimizer, loss=loss, metrics=metrics
            )

            assert model.optimizer
            assert model.loss
            assert model._compiled
            assert model.metric_names == ['mock_metric']
            assert model.metric_fns == [mock_metric]

            # reset for next iteration
            model.optimizer = None
            model.loss = None
            model._compiled = False
            model.metric_names = []
            model.metric_fns = []

        with pytest.raises(AttributeError):
            model.compile(self=model, optimizer='BadOptimizer', loss='BCELoss')

        with pytest.raises(AttributeError):
            model.compile(self=model, optimizer='Adam', loss='BadLoss')

    def test_evaluate_generator(self):
        """Test evaluate_generator"""

        model = MagicMock()
        model.network = MagicMock()
        model.metric_fns = ['sentinel_metric']
        model.device = MagicMock()
        model.evaluate_generator = Model.evaluate_generator

        def test_on_batch(inputs, targets):
            """Mock test_on_batch

            `inputs` will consist of an array with a single value, which will
            be used to build the output of `test_on_batch`.
            """

            unique_values = torch.unique(inputs)
            assert len(unique_values) == 1
            input_value = unique_values[0].tolist()
            return (input_value, input_value * 2)
        model.test_on_batch = test_on_batch

        def generator():
            """Mock generator function"""

            n_obs = 1
            while True:
                inputs = torch.ones((n_obs, 64, 64, 3)) * n_obs
                targets = torch.ones((n_obs, 1)) * n_obs
                n_obs += 1

                yield (inputs, targets)

        test_cases = [
            {'n_steps': 10, 'device': 'cpu',
             'expected_loss': 7, 'expected_metric_value': 14},
            {'n_steps': 58, 'expected_loss': 39,
             'expected_metric_value': 78}
        ]

        for test_case in test_cases:
            n_steps = test_case['n_steps']
            device = test_case.get('device')

            model.device = device
            val_outputs = model.evaluate_generator(
                self=model, generator=generator(), n_steps=n_steps
            )

            assert np.allclose(
                val_outputs[0], test_case['expected_loss'], atol=1e-4
            )
            assert np.allclose(
                val_outputs[1], test_case['expected_metric_value'],
                atol=1e-4
            )
            assert model._assert_compiled.call_count == 1

            # re-assign before the next iteration of the loop
            model._assert_compiled.call_count = 0

    def test_fit_generator(self, monkeypatch):
        """Test fit_generator method

        This tests that the correct total number of steps are taken for a given
        `fit_generator` call with a specified `n_steps_per_epoch` and
        `n_epochs`.
        """

        model = MagicMock()
        model.stop_training = False
        model.network = MagicMock()
        model.train_on_batch = MagicMock()
        model.train_on_batch.return_value = (4, 5)
        model.device = MagicMock()
        model.fit_generator = Model.fit_generator
        model.evaluate_generator = MagicMock()
        model.evaluate_generator.return_value = (2, 3)
        model.metric_names = ['mock_metric']

        generator = MagicMock()
        inputs = torch.randn((2, 64, 64, 3))
        targets = torch.randn((2, 1))
        generator.__next__ = MagicMock()
        generator.__next__.return_value = (inputs, targets)

        test_cases = [
            {'n_steps_per_epoch': 1, 'n_epochs': 1, 'device': 'cpu'},
            {'n_steps_per_epoch': 2, 'n_epochs': 2,
             'validation_data': generator, 'n_validation_steps': 10},
            {'n_steps_per_epoch': 2, 'n_epochs': 2,
             'validation_data': generator, 'n_validation_steps': 10,
             'early_stopping': True},
            {'n_steps_per_epoch': 223, 'n_epochs': 50, 'device': 'cpu'}
        ]

        for test_case in test_cases:
            default_callbacks = MagicMock()
            default_callbacks.return_value = [1, 2, 3]
            model._default_callbacks = default_callbacks
            early_stopping = test_case.get('early_stopping', False)
            if early_stopping:
                model.stop_training = True

            n_steps_per_epoch = test_case['n_steps_per_epoch']
            n_epochs = test_case['n_epochs']
            device = test_case.get('device')
            validation_data = test_case.get('validation_data')
            n_validation_steps = test_case.get('n_validation_steps')

            mock_callback_list = MagicMock()
            mock_callbacks = create_autospec(CallbackList)
            mock_callback_list.return_value = mock_callbacks
            monkeypatch.setattr(
                'ktorch.model.CallbackList', mock_callback_list
            )
            mock_progbar = MagicMock()
            mock_progbar.return_value = 7
            monkeypatch.setattr(
                'ktorch.model.ProgbarLogger', mock_progbar
            )

            model.device = device
            model.fit_generator(
                self=model, generator=generator,
                n_steps_per_epoch=n_steps_per_epoch, n_epochs=n_epochs,
                validation_data=validation_data,
                n_validation_steps=n_validation_steps,
                callbacks=[4, 5]
            )
            assert model._assert_compiled.call_count == 1
            assert mock_callbacks.on_train_begin.call_count == 1
            assert mock_callbacks.on_train_end.call_count == 1
            if not early_stopping:
                n_batches = n_steps_per_epoch * n_epochs
                assert model.train_on_batch.call_count == n_batches
                model.train_on_batch.assert_called_with(inputs, targets)
                assert generator.__next__.call_count == n_batches
                assert mock_callbacks.on_epoch_begin.call_count == n_epochs
                assert mock_callbacks.on_batch_begin.call_count == n_batches
                assert mock_callbacks.on_batch_end.call_count == n_batches
                mock_callbacks.on_batch_end.assert_any_call(
                    0, {'batch': 0, 'size': 1, 'loss': 4, 'mock_metric': 5}
                )

            mock_callback_list.assert_called_with([1, 2, 3, 7, 4, 5])
            if validation_data is not None:
                expected_metrics = [
                    'loss', 'val_loss', 'mock_metric', 'val_mock_metric'
                ]
            else:
                expected_metrics = ['loss', 'mock_metric']
            mock_callbacks.set_params.assert_called_with(
                {'epochs': n_epochs,
                 'metrics': expected_metrics,
                 'steps': n_steps_per_epoch, 'verbose': True}
            )
            mock_callbacks.set_model.assert_called_with(model)

            epoch_logs = {}
            if validation_data is not None:
                model.evaluate_generator.assert_called_with(
                    validation_data, n_validation_steps
                )
                epoch_logs['val_loss'] = 2
                epoch_logs['val_mock_metric'] = 3

            if not early_stopping:
                mock_callbacks.on_epoch_end.assert_any_call(0, epoch_logs)

            # reset the call counts for the next iteration
            model._assert_compiled.call_count = 0
            model.train_on_batch.call_count = 0
            model.evaluate_generator.call_count = 0
            model.stop_training = False
            generator.__next__.call_count = 0

    def test_fit_generator__bad_input(self):
        """Test fit_generator method with bad inputs

        `fit_generator` throws a RuntimeError if only one of `validation_data`
        and `n_validation_steps` are passed in.
        """

        model = MagicMock()
        model._assert_compiled = MagicMock()
        model.fit_generator = Model.fit_generator

        with pytest.raises(RuntimeError):
            model.fit_generator(
                self=model, generator='sentinel1',
                n_steps_per_epoch='sentinel2', n_epochs='sentinel3',
                validation_data='sentinel4'
            )

        with pytest.raises(RuntimeError):
            model.fit_generator(
                self=model, generator='sentinel1',
                n_steps_per_epoch='sentinel2', n_epochs='sentinel3',
                n_validation_steps='sentinel4'
            )

    def test_load_weights(self, monkeypatch):
        """Test load_weights method

        This tests two things:
        - That `torch.load` is called as expected; it mocks this function to
          prevent anything from actually being loaded from disk
        - That the `Model.network.load_state_dict` method is called
        """

        model = MagicMock()
        model.network = MagicMock()
        load_state_dict_mock = MagicMock()
        model.network.load_state_dict = load_state_dict_mock
        model.load_weights = Model.load_weights

        mocked_load = MagicMock()
        mocked_load.return_value = 'return_from_torch_load'
        monkeypatch.setattr(torch, 'load', mocked_load)

        model.load_weights(
            self=model, fpath_weights='/home/sallamander/weights.pt'
        )
        mocked_load.assert_called_once_with('/home/sallamander/weights.pt')
        load_state_dict_mock.assert_called_once_with('return_from_torch_load')

    def test_save_weights(self):
        """Test save_weights method

        This tests two things:
        - That `torch.save` is called as expected; it mocks this function to
          prevent anything from actually being saved to disk
        - That an `AssertionError` is thrown if `save_weights` is called with
          `overwrite=False` (which is not currently supported)
        """

        model = MagicMock()
        model.network = MagicMock()
        model.network.state_dict = MagicMock()
        model.network.state_dict.return_value = {'state': 'dict'}
        model.save_weights = Model.save_weights

        with pytest.raises(AssertionError):
            model.save_weights(
                self=model, fpath_weights='sentinel', overwrite=False
            )

        with patch.object(torch, 'save', wraps=torch.save) as patched_save:
            model.save_weights(
                self=model, fpath_weights='/home/sallamander/weights.pt'
            )
            patched_save.assert_called_once_with(
                {'state': 'dict'}, '/home/sallamander/weights.pt'
            )
            assert model.network.state_dict.call_count == 1

    def test_test_on_batch(self):
        """Test test_on_batch method"""

        model = create_autospec(Model)
        model.device = 'cpu'
        model.test_on_batch = Model.test_on_batch
        model.network = MagicMock()
        model.network.train = MagicMock()

        mock_metric = MagicMock()
        mock_metric.return_value = 4
        model.metric_fns = [mock_metric]

        inputs = torch.randn((2, 3), requires_grad=True)
        targets = torch.randint(size=(2,), high=2, dtype=torch.int64)
        outputs = torch.nn.Sigmoid()(inputs)

        model.loss = MagicMock()
        loss_value = torch.nn.CrossEntropyLoss()(outputs, targets)
        model.loss.return_value = loss_value

        test_outputs = model.test_on_batch(
            self=model, inputs=inputs, targets=targets
        )

        assert test_outputs[0] == loss_value.tolist()
        assert test_outputs[1] == 4

        assert mock_metric.call_count == 1
        assert model.network.call_count == 1
        assert model._assert_compiled.call_count == 1
        model.network.train.assert_called_with(mode=False)
        assert model.loss.call_count == 1

    def test_train_on_batch(self):
        """Test train_on_batch method"""

        model = create_autospec(Model)
        model.device = 'cpu'
        model.train_on_batch = Model.train_on_batch
        model.network = MagicMock()
        model.network.train = MagicMock()

        mock_metric = MagicMock()
        mock_metric.return_value = 4
        model.metric_fns = [mock_metric]

        inputs = torch.randn((2, 3), requires_grad=True)
        targets = torch.randint(size=(2,), high=2, dtype=torch.int64)
        outputs = torch.nn.Sigmoid()(inputs)

        model.loss = MagicMock()
        loss_value = torch.nn.CrossEntropyLoss()(outputs, targets)
        model.loss.return_value = loss_value
        model.optimizer = create_autospec(torch.optim.Adam)

        with patch.object(loss_value, 'backward') as patched_backward:
            train_outputs = model.train_on_batch(
                self=model, inputs=inputs, targets=targets
            )

        assert train_outputs[0] == loss_value.tolist()
        assert train_outputs[1] == 4

        assert mock_metric.call_count == 1
        assert model.network.call_count == 1
        model.network.train.assert_called_with(mode=True)
        assert model.loss.call_count == 1
        assert model.optimizer.zero_grad.call_count == 1
        assert model.optimizer.step.call_count == 1
        assert patched_backward.call_count == 1
