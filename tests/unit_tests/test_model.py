"""Unit tests for model.py"""

from itertools import product
from unittest.mock import create_autospec, patch, MagicMock

import numpy as np
import pytest
import torch
from keras.callbacks import (
    BaseLogger, CallbackList, History, ProgbarLogger
)
from torch.tensor import Tensor
from torch._C import _TensorBase
from torch.nn import BCELoss, CrossEntropyLoss, L1Loss
from torch.optim import Adam, RMSprop

from ktorch.model import Model


class TestModel():
    """Tests for Model with a single input / output network"""

    def _check_train_or_test_on_batch(self, check):
        """Assert the output from train_on_batch or test_on_batch

        :param check: what method to test, one of 'train' or 'test'
        :type check: str
        """

        if check == 'train':
            call_fn = Model.train_on_batch
        else:
            call_fn = Model.test_on_batch

        model = create_autospec(Model)
        model.device = 'cpu'
        model.network = MagicMock()

        mock_metric = MagicMock()
        mock_metric.return_value = 4
        model.metric_fns = [mock_metric]

        inputs = MagicMock()
        inputs.to = MagicMock()
        input_values = torch.randn((2, 3), requires_grad=True)
        inputs.to.return_value = input_values

        targets = MagicMock()
        targets.to = MagicMock()
        target_values = (
            torch.randint(size=(2,), high=2, dtype=torch.int64)
        )
        targets.to.return_value = target_values
        outputs = torch.nn.Sigmoid()(input_values)

        model.loss = MagicMock()
        loss_value = torch.nn.CrossEntropyLoss()(outputs, target_values)
        model.loss.return_value = loss_value
        model.optimizer = create_autospec(torch.optim.Adam)

        with patch.object(loss_value, 'backward') as patched_backward:
            outputs = call_fn(self=model, inputs=inputs, targets=targets)

        inputs.to.assert_called_once_with('cpu')
        targets.to.assert_called_once_with('cpu')

        assert outputs[0] == loss_value.tolist()
        assert outputs[1] == 4

        assert mock_metric.call_count == 1
        assert model.network.call_count == 1
        assert model.loss.call_count == 1
        assert model._assert_compiled.call_count == 1

        if check == 'train':
            model.network.train.assert_called_with(mode=True)
            assert model.optimizer.zero_grad.call_count == 1
            assert model.optimizer.step.call_count == 1
            assert patched_backward.call_count == 1
        else:
            model.network.train.assert_called_with(mode=False)
            assert model.optimizer.zero_grad.call_count == 0
            assert model.optimizer.step.call_count == 0
            assert patched_backward.call_count == 0

    def test_init(self):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.
        """

        network = MagicMock()
        gpu_id = MagicMock()

        model = Model(network, n_outputs=1, gpu_id=gpu_id)
        assert id(network) == id(model.network)
        assert id(gpu_id) == id(model.gpu_id)
        assert model.n_outputs == 1
        assert not model._compiled
        assert not model.optimizer
        assert not model.loss
        assert model.history
        assert isinstance(model.history, History)
        assert not model.stop_training
        assert not model.metric_names
        assert not model.metric_fns

        model = Model(network, n_outputs=1)
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
        model.n_outputs = 1
        model.optimizer = None
        model.loss = None
        model._compiled = False
        model.compile = Model.compile
        model.metric_names = []
        model.metric_fns = []

        network = MagicMock()
        parameters_fn = MagicMock()
        mock_parameters = [torch.nn.Parameter(torch.randn((64, 64)))]
        parameters_fn.return_value = mock_parameters
        network.parameters = parameters_fn
        model.network = network

        valid_optimizers = [
            'Adam', 'RMSprop',
            Adam(params=mock_parameters, lr=1e-4),
            RMSprop(params=mock_parameters, lr=1e-5)
        ]
        valid_losses = [
            'BCELoss', 'CrossEntropyLoss', 'L1Loss',
            BCELoss(), CrossEntropyLoss(), L1Loss()
        ]

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

    def test_evaluate(self):
        """Test evaluate"""

        model = MagicMock()
        model.n_outputs = 1
        model.network = MagicMock()
        model.metric_fns = ['sentinel_metric']
        model.device = MagicMock()
        model.evaluate = Model.evaluate

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

        test_cases = [
            {'n_batches': 10, 'device': 'cpu',
             'expected_loss': 5.5, 'expected_metric_value': 11.0},
            {'n_batches': 58, 'expected_loss': 29.5,
             'expected_metric_value': 59.0}
        ]

        for test_case in test_cases:
            n_batches = test_case['n_batches']
            inputs = []
            for idx in range(1, n_batches + 1):
                inputs.append(torch.ones((2, 64, 64, 3)) * idx)
            inputs = torch.cat(inputs)
            targets = []
            for idx in range(1, n_batches + 1):
                targets.append(torch.ones((2, 1)) * idx)
            targets = torch.cat(targets)

            device = test_case.get('device')

            model.device = device
            val_outputs = model.evaluate(
                self=model, x=inputs, y=targets, batch_size=2
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

    def test_fit(self, monkeypatch):
        """Test fit method

        This tests that the correct total number of steps are taken for a given
        `fit` call with a specified `x`, `y`, `batch_size`, and `n_epochs`.
        """

        model = MagicMock()
        model.n_outputs = 1
        model.stop_training = False
        model.network = MagicMock()
        model.train_on_batch = MagicMock()
        model.train_on_batch.return_value = (4, 5)
        model.device = MagicMock()
        model.fit = Model.fit
        model.evaluate = MagicMock()
        model.evaluate.return_value = (2, 3)
        model.metric_names = ['mock_metric']

        generator = MagicMock()
        inputs_template = torch.randn((2, 64, 64, 3))
        targets_template = torch.randn((2, 1))

        test_cases = [
            {'n_batches_per_epoch': 1, 'n_epochs': 1, 'device': 'cpu'},
            {'n_batches_per_epoch': 2, 'n_epochs': 2,
             'use_validation_data': True},
            {'n_batches_per_epoch': 2, 'n_epochs': 2,
             'use_validation_data': True, 'early_stopping': True},
            {'n_batches_per_epoch': 223, 'n_epochs': 50, 'device': 'cpu'}
        ]

        for test_case in test_cases:
            default_callbacks = MagicMock()
            default_callbacks.return_value = [1, 2, 3]
            model._default_callbacks = default_callbacks
            early_stopping = test_case.get('early_stopping', False)
            if early_stopping:
                model.stop_training = True

            n_batches_per_epoch = test_case['n_batches_per_epoch']
            inputs = torch.cat([inputs_template] * n_batches_per_epoch)
            targets = torch.cat([targets_template] * n_batches_per_epoch)

            n_epochs = test_case['n_epochs']
            device = test_case.get('device')
            use_validation_data = test_case.get('use_validation_data', False)
            if use_validation_data:
                validation_data = (inputs, targets)
            else:
                validation_data = None

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
            model.fit(
                self=model, x=inputs, y=targets, batch_size=2,
                n_epochs=n_epochs, validation_data=validation_data,
                callbacks=[4, 5]
            )
            assert model._assert_compiled.call_count == 1
            assert mock_callbacks.on_train_begin.call_count == 1
            assert mock_callbacks.on_train_end.call_count == 1
            if not early_stopping:
                n_total_batches = n_batches_per_epoch * n_epochs
                assert model.train_on_batch.call_count == n_total_batches
                assert mock_callbacks.on_epoch_begin.call_count == n_epochs
                assert (
                    mock_callbacks.on_batch_begin.call_count == n_total_batches
                )
                assert (
                    mock_callbacks.on_batch_end.call_count == n_total_batches
                )
                mock_callbacks.on_batch_end.assert_any_call(
                    0, {'batch': 0, 'size': 2, 'loss': 4, 'mock_metric': 5}
                )

            mock_callback_list.assert_called_with([1, 2, 3, 7, 4, 5])
            if validation_data is not None:
                expected_metrics = [
                    'loss', 'val_loss', 'mock_metric', 'val_mock_metric'
                ]
            else:
                expected_metrics = ['loss', 'mock_metric']
            mock_callbacks.set_params.assert_called_with(
                {'batch_size': 2, 'epochs': n_epochs,
                 'metrics': expected_metrics,
                 'steps': None, 'verbose': True,
                 'samples': inputs.shape[0]}
            )
            mock_callbacks.set_model.assert_called_with(model)

            epoch_logs = {}
            if validation_data is not None:
                if not early_stopping:
                    assert model.evaluate.call_count == n_epochs
                epoch_logs['val_loss'] = 2
                epoch_logs['val_mock_metric'] = 3

            if not early_stopping:
                mock_callbacks.on_epoch_end.assert_any_call(0, epoch_logs)

            # reset the call counts for the next iteration
            model._assert_compiled.call_count = 0
            model.train_on_batch.call_count = 0
            model.evaluate.call_count = 0
            model.stop_training = False

    def test_fit_generator(self, monkeypatch):
        """Test fit_generator method

        This tests that the correct total number of steps are taken for a given
        `fit_generator` call with a specified `n_steps_per_epoch` and
        `n_epochs`.
        """

        model = MagicMock()
        model.n_outputs = 1
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
                n_batches_per_epoch = n_steps_per_epoch * n_epochs
                assert model.train_on_batch.call_count == n_batches_per_epoch
                model.train_on_batch.assert_called_with(inputs, targets)
                assert generator.__next__.call_count == n_batches_per_epoch
                assert mock_callbacks.on_epoch_begin.call_count == n_epochs
                assert (
                    mock_callbacks.on_batch_begin.call_count ==
                    n_batches_per_epoch
                )
                assert (
                    mock_callbacks.on_batch_end.call_count ==
                    n_batches_per_epoch
                )
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

        self._check_train_or_test_on_batch(check='test')

    def test_train_on_batch(self):
        """Test train_on_batch method"""

        self._check_train_or_test_on_batch(check='train')


class TestModel__MultiOutput():
    """Tests for Model with a single input, multiple output network"""

    def _check_train_or_test_on_batch(self, check):
        """Assert the output from train_on_batch or test_on_batch

        :param check: what method to test, one of 'train' or 'test'
        :type check: str
        """

        model = create_autospec(Model)

        if check == 'train':
            call_fn = Model.train_on_batch
        else:
            call_fn = Model.test_on_batch

        model = create_autospec(Model)
        model.device = 'cpu'
        model.network = MagicMock()

        inputs = MagicMock()
        inputs.to = MagicMock()
        inputs.to.return_value = torch.randn((2, 3), requires_grad=True)

        target1 = MagicMock()
        target1.to = MagicMock()
        target1.to.return_value = (
            torch.ones((2, ), requires_grad=True, dtype=torch.float32) * 0.25
        )
        target2 = MagicMock()
        target2.to = MagicMock()
        target2.to.return_value = (
            torch.ones((2, ), requires_grad=True, dtype=torch.float32) * 2
        )
        targets = [target1, target2]
        outputs = (
            torch.ones((2, ), requires_grad=True, dtype=torch.float32),
            torch.ones((2, ), requires_grad=True, dtype=torch.float32)
        )
        model.network.return_value = outputs

        mock_metric1 = MagicMock()
        mock_metric1.side_effect = (
            lambda y_true, y_pred: (torch.mean(y_true - y_pred) + 0.5).item()
        )
        mock_metric2 = MagicMock()
        mock_metric2.side_effect = (
            lambda y_true, y_pred: (torch.mean(y_true - y_pred) * 2).item()
        )
        model.metric_fns = [mock_metric1, mock_metric2]

        loss = create_autospec(torch.Tensor)
        loss.side_effect = (
            lambda outputs, targets: (torch.mean(outputs - targets) * 3)
        )
        model.loss = loss
        model.optimizer = create_autospec(torch.optim.Adam)

        with patch.object(loss, 'backward') as patched_backward:
            outputs = call_fn(self=model, inputs=inputs, targets=targets)

        inputs.to.assert_called_once_with('cpu')
        target1.to.assert_called_once_with('cpu')
        target2.to.assert_called_once_with('cpu')

        assert model.network.call_count == 1
        assert loss.call_count == 2
        assert mock_metric1.call_count == 2
        assert mock_metric2.call_count == 2

        assert model._assert_compiled.call_count
        assert np.allclose(
            outputs, [-0.75, 2.25, -3.0, -0.25, 1.5, -1.5, 2.0]
        )

        if check == 'train':
            model.network.train.assert_called_once_with(mode=True)
            assert model.optimizer.zero_grad.call_count == 1
            assert model.optimizer.step.call_count == 1
            # backward is not called on the mocked loss, since there are
            # multiple outputs and the backward is called on the sum of the
            # multiple outputs
            assert patched_backward.call_count == 0
        else:
            model.network.train.assert_called_once_with(mode=False)
            assert model.optimizer.zero_grad.call_count == 0
            assert model.optimizer.step.call_count == 0
            assert patched_backward.call_count == 0

    def test_compile(self):
        """Test compile method

        This tests mainly tests the parts of the `Model.compile` method that
        should lead to different results when there are multiple outputs (i.e.
        `Model.n_outputs > 1`) instead of a single one. The only thing that
        changes in the `compile` method is that the `Model.metric_names`
        attribute should include a metric name for each metric for each output.

        It asserts a handful of other things to try to ensure that there are no
        regressions from expected behavior when there are multiple outputs.
        """

        model = MagicMock()
        model.optimizer = None
        model.loss = None
        model._compiled = False
        model.n_outputs = 2
        model.metric_names = []
        model.metric_fns = []
        model.compile = Model.compile

        metric1 = MagicMock()
        metric1.name = 'metric1'
        metric2 = MagicMock()
        metric2.name = 'metric2'
        metrics = [metric1, metric2]

        mock_parameters = [torch.nn.Parameter(torch.randn((64, 64)))]
        optimizer = Adam(params=mock_parameters, lr=1e-4),
        loss = BCELoss()

        assert not model.optimizer
        assert not model.loss
        assert not model._compiled

        model.compile(
            self=model, optimizer=optimizer, loss=loss, metrics=metrics
        )

        assert model.optimizer
        assert model.loss
        assert model._compiled
        assert model.metric_names == (
            ['metric11', 'metric12', 'metric21', 'metric22']
        )
        assert model.metric_fns == metrics

    def test_evaluate(self):
        """Test evaluate"""

        model = MagicMock()
        model.n_outputs = 2
        model.network = MagicMock()
        model.metric_fns = ['sentinel_metric']
        model.device = MagicMock()
        model.evaluate = Model.evaluate

        def test_on_batch(inputs, targets):
            """Mock test_on_batch

            `inputs` will consist of an array with a single value, which will
            be used to build the output of `test_on_batch`.
            """

            unique_values = torch.unique(inputs)
            assert len(unique_values) == 1
            input_value = unique_values[0].tolist()
            return (
                input_value, input_value - 1, input_value + 1,
                input_value * 2, input_value * 3
            )
        model.test_on_batch = test_on_batch

        test_cases = [
            {'n_batches': 10, 'device': 'cpu',
             'expected_outputs': [5.5, 4.5, 6.5, 11.0, 16.5]},
            {'n_batches': 58,
             'expected_outputs': [29.5, 28.5, 30.5, 59.0, 88.5]}
        ]

        for test_case in test_cases:
            n_batches = test_case['n_batches']
            inputs = []
            for idx in range(1, n_batches + 1):
                inputs.append(torch.ones((2, 64, 64, 3)) * idx)
            inputs = torch.cat(inputs)
            targets1 = []
            targets2 = []
            for idx in range(1, n_batches + 1):
                targets1.append(torch.ones((2, 1)) * idx)
                targets2.append(torch.ones((2, 1)) * idx)
            targets1 = torch.cat(targets1)
            targets2 = torch.cat(targets2)
            targets = [targets1, targets2]

            device = test_case.get('device')

            model.device = device
            val_outputs = model.evaluate(
                self=model, x=inputs, y=targets, batch_size=2
            )

            assert np.allclose(
                val_outputs, test_case['expected_outputs'], atol=1e-4
            )
            assert model._assert_compiled.call_count == 1

            # re-assign before the next iteration of the loop
            model._assert_compiled.call_count = 0

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
            return (
                input_value, input_value - 1, input_value + 1,
                input_value * 2, input_value * 3
            )
        model.test_on_batch = test_on_batch

        def generator():
            """Mock generator function"""

            n_obs = 1
            while True:
                inputs = torch.ones((n_obs, 64, 64, 3)) * n_obs
                targets = torch.ones((n_obs, 1)) * n_obs
                n_obs += 1

                yield (inputs, [targets, targets + 1])

        test_cases = [
            {'n_steps': 10, 'device': 'cpu',
             'expected_outputs': [7.0, 6.0, 8.0, 14.0, 21.0]},
            {'n_steps': 58,
             'expected_outputs': [39.0, 38.0, 40.0, 78.0, 117.0]}
        ]

        for test_case in test_cases:
            n_steps = test_case['n_steps']
            device = test_case.get('device')

            model.device = device
            val_outputs = model.evaluate_generator(
                self=model, generator=generator(), n_steps=n_steps
            )

            assert np.allclose(
                val_outputs, test_case['expected_outputs'], atol=1e-4
            )
            assert model._assert_compiled.call_count == 1

            # re-assign before the next iteration of the loop
            model._assert_compiled.call_count = 0

    def test_fit(self, monkeypatch):
        """Test fit method

        This tests that the correct total number of steps are taken for a given
        `fit` call with a specified `x`, `y`, `batch_size`, and `n_epochs`.
        """

        model = MagicMock()
        model.n_outputs = 2
        model.stop_training = False
        model.network = MagicMock()
        model.train_on_batch = MagicMock()
        model.train_on_batch.return_value = (4, 1, 3, 5, 2)
        model.device = MagicMock()
        model.fit = Model.fit
        model.evaluate = MagicMock()
        model.evaluate.return_value = (2, 1.5, 0.5, 3, 1)
        model.metric_names = ['mock_metric1', 'mock_metric2']

        generator = MagicMock()
        inputs_template = torch.randn((2, 64, 64, 3))
        targets_template = torch.randn((2, 1))

        test_cases = [
            {'n_batches_per_epoch': 1, 'n_epochs': 1, 'device': 'cpu'},
            {'n_batches_per_epoch': 2, 'n_epochs': 2,
             'use_validation_data': True},
            {'n_batches_per_epoch': 2, 'n_epochs': 2,
             'use_validation_data': True, 'early_stopping': True},
            {'n_batches_per_epoch': 223, 'n_epochs': 50, 'device': 'cpu'}
        ]

        for test_case in test_cases:
            default_callbacks = MagicMock()
            default_callbacks.return_value = [1, 2, 3]
            model._default_callbacks = default_callbacks
            early_stopping = test_case.get('early_stopping', False)
            if early_stopping:
                model.stop_training = True

            n_batches_per_epoch = test_case['n_batches_per_epoch']
            inputs = torch.cat([inputs_template] * n_batches_per_epoch)
            targets = [torch.cat([targets_template] * n_batches_per_epoch)] * 2

            n_epochs = test_case['n_epochs']
            device = test_case.get('device')
            use_validation_data = test_case.get('use_validation_data', False)
            if use_validation_data:
                validation_data = (inputs, targets)
            else:
                validation_data = None

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
            model.fit(
                self=model, x=inputs, y=targets, batch_size=2,
                n_epochs=n_epochs, validation_data=validation_data,
                callbacks=[4, 5]
            )
            assert model._assert_compiled.call_count == 1
            assert mock_callbacks.on_train_begin.call_count == 1
            assert mock_callbacks.on_train_end.call_count == 1
            if not early_stopping:
                n_total_batches = n_batches_per_epoch * n_epochs
                assert model.train_on_batch.call_count == n_total_batches
                assert mock_callbacks.on_epoch_begin.call_count == n_epochs
                assert (
                    mock_callbacks.on_batch_begin.call_count == n_total_batches
                )
                assert (
                    mock_callbacks.on_batch_end.call_count == n_total_batches
                )
                mock_callbacks.on_batch_end.assert_any_call(
                    0,
                    {'batch': 0, 'size': 2, 'loss': 4, 'loss1': 1, 'loss2': 3,
                     'mock_metric1': 5, 'mock_metric2': 2}
                )

            mock_callback_list.assert_called_with([1, 2, 3, 7, 4, 5])
            if validation_data is not None:
                expected_metrics = [
                    'loss', 'loss1', 'loss2', 'val_loss', 'val_loss1',
                    'val_loss2', 'mock_metric1', 'val_mock_metric1',
                    'mock_metric2', 'val_mock_metric2'
                ]
            else:
                expected_metrics = [
                    'loss', 'loss1', 'loss2', 'mock_metric1', 'mock_metric2'
                ]
            mock_callbacks.set_params.assert_called_with(
                {'batch_size': 2, 'epochs': n_epochs,
                 'metrics': expected_metrics,
                 'steps': None, 'verbose': True,
                 'samples': inputs.shape[0]}
            )
            mock_callbacks.set_model.assert_called_with(model)

            epoch_logs = {}
            if validation_data is not None:
                if not early_stopping:
                    assert model.evaluate.call_count == n_epochs
                epoch_logs['val_loss'] = 2
                epoch_logs['val_loss1'] = 1.5
                epoch_logs['val_loss2'] = 0.5
                epoch_logs['val_mock_metric1'] = 3
                epoch_logs['val_mock_metric2'] = 1

            if not early_stopping:
                mock_callbacks.on_epoch_end.assert_any_call(0, epoch_logs)

            # reset the call counts for the next iteration
            model._assert_compiled.call_count = 0
            model.train_on_batch.call_count = 0
            model.evaluate.call_count = 0
            model.stop_training = False

    def test_fit_generator(self, monkeypatch):
        """Test fit_generator method

        This tests that the correct total number of steps are taken for a given
        `fit` call with a specified `x`, `y`, `batch_size`, and `n_epochs`. It
        also tests that certain objects internal to the `fit` method (e.g. the
        callbacks object) are called as expected.
        """

        model = MagicMock()
        model.n_outputs = 2
        model.stop_training = False
        model.network = MagicMock()
        model.train_on_batch = MagicMock()
        model.train_on_batch.return_value = (4, 1, 3, 5, 2)
        model.device = MagicMock()
        model.fit_generator = Model.fit_generator
        model.evaluate_generator = MagicMock()
        model.evaluate_generator.return_value = (2, 1.5, 0.5, 3, 1)
        model.metric_names = ['mock_metric1', 'mock_metric2']

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
                n_batches_per_epoch = n_steps_per_epoch * n_epochs
                assert model.train_on_batch.call_count == n_batches_per_epoch
                model.train_on_batch.assert_called_with(inputs, targets)
                assert generator.__next__.call_count == n_batches_per_epoch
                assert mock_callbacks.on_epoch_begin.call_count == n_epochs
                assert (
                    mock_callbacks.on_batch_begin.call_count ==
                    n_batches_per_epoch
                )
                assert (
                    mock_callbacks.on_batch_end.call_count ==
                    n_batches_per_epoch
                )
                mock_callbacks.on_batch_end.assert_any_call(
                    0,
                    {'batch': 0, 'size': 1, 'loss': 4, 'loss1': 1, 'loss2': 3,
                     'mock_metric1': 5, 'mock_metric2': 2}
                )

            mock_callback_list.assert_called_with([1, 2, 3, 7, 4, 5])
            if validation_data is not None:
                expected_metrics = [
                    'loss', 'loss1', 'loss2', 'val_loss', 'val_loss1',
                    'val_loss2', 'mock_metric1', 'val_mock_metric1',
                    'mock_metric2', 'val_mock_metric2'
                ]
            else:
                expected_metrics = [
                    'loss', 'loss1', 'loss2', 'mock_metric1', 'mock_metric2'
                ]
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
                epoch_logs['val_loss1'] = 1.5
                epoch_logs['val_loss2'] = 0.5
                epoch_logs['val_mock_metric1'] = 3
                epoch_logs['val_mock_metric2'] = 1

            if not early_stopping:
                mock_callbacks.on_epoch_end.assert_any_call(0, epoch_logs)

            # reset the call counts for the next iteration
            model._assert_compiled.call_count = 0
            model.train_on_batch.call_count = 0
            model.evaluate_generator.call_count = 0
            model.stop_training = False
            generator.__next__.call_count = 0

    def test_test_on_batch(self):
        """Test test_on_batch method"""

        self._check_train_or_test_on_batch(check='test')

    def test_train_on_batch(self):
        """Test train_on_batch method"""

        self._check_train_or_test_on_batch(check='train')
