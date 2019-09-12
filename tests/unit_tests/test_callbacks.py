"""Tests for callbacks.py"""

import os
from itertools import product
from unittest.mock import call, MagicMock

import numpy as np
import pytest
from torch.utils.tensorboard.writer import FileWriter

from ktorch.callbacks import LRFinder, TensorBoard


class TestLRFinder():
    """Tests for LRFinder"""

    def test_init(self):
        """Test __init__"""

        lr_finder = LRFinder(
            dirpath_results='dirpath_results', n_steps_per_epoch=5
        )

        assert lr_finder.min_lr == 1e-5
        assert lr_finder.max_lr == 1e-2
        assert lr_finder.dirpath_results == 'dirpath_results'
        assert lr_finder.n_total_iterations == 5
        assert lr_finder.iteration == 0
        assert lr_finder.history == {}
        assert lr_finder.df_history is None
        assert lr_finder.model is None

    def test_calc_learning_rate(self):
        """Test _calc_learning_rate"""

        lr_finder = MagicMock()
        lr_finder._calc_learning_rate = LRFinder._calc_learning_rate

        lr_finder.min_lr = 1e-5
        lr_finder.max_lr = 1e-2
        lr_finder.n_total_iterations = 125

        iterations = [1, 5, 10, 25, 100]
        expected_learning_rates = [
            8.992e-05, 0.0004096, 0.000809, 0.002008, 0.008002
        ]
        it = zip(iterations, expected_learning_rates)
        for iteration, expected_learning_rate in it:
            lr_finder.iteration = iteration

            learning_rate = lr_finder._calc_learning_rate(self=lr_finder)
            assert np.allclose(
                learning_rate, expected_learning_rate, atol=1e-6
            )

    def test_on_batch_end(self):
        """Test on_batch_end"""

        lr_finder = MagicMock()
        lr_finder.on_batch_end = LRFinder.on_batch_end
        lr_finder.alpha = 0.98
        lr_finder.iteration = 0
        lr_finder.history = {}

        optimizer = MagicMock
        optimizer.param_groups = [{'lr': 0.00001}, {'lr': 0.00001}]
        lr_finder.optimizer = optimizer

        calc_learning_rate_fn = MagicMock()
        calc_learning_rate_fn.return_value = 0.0001
        lr_finder._calc_learning_rate = calc_learning_rate_fn

        lr_finder.on_batch_end(self=lr_finder, _=None, logs={'loss': 1})
        assert lr_finder.iteration == 1
        assert lr_finder.history['lr'] == [0.00001]
        assert lr_finder.history['loss'] == [1]
        assert lr_finder.history['iterations'] == [1]
        assert np.allclose(
            lr_finder.history['smoothed_loss'], [48.999999]
        )
        assert (
            set(lr_finder.history.keys()) ==
            {'iterations', 'lr', 'loss', 'smoothed_loss'}
        )
        assert optimizer.param_groups[0]['lr'] == 0.0001
        assert optimizer.param_groups[1]['lr'] == 0.0001

        lr_finder.on_batch_end(self=lr_finder, _=None, logs={'loss': 1})
        assert lr_finder.iteration == 2
        assert lr_finder.history['lr'] == [0.00001, 0.0001]
        assert lr_finder.history['loss'] == [1, 1]
        assert lr_finder.history['iterations'] == [1, 2]
        assert np.allclose(
            lr_finder.history['smoothed_loss'], [48.999999, 49.49494949]
        )
        assert (
            set(lr_finder.history.keys()) ==
            {'iterations', 'lr', 'loss', 'smoothed_loss'}
        )
        assert optimizer.param_groups[0]['lr'] == 0.0001
        assert optimizer.param_groups[1]['lr'] == 0.0001

    def test_on_train_begin(self):
        """Test on_train_begin"""

        lr_finder = MagicMock()
        lr_finder.on_train_begin = LRFinder.on_train_begin

        optimizer = MagicMock
        optimizer.param_groups = [{'lr': 1}, {'lr': 1}]
        lr_finder.optimizer = optimizer

        lr_finder.min_lr = 0.001
        lr_finder.on_train_begin(self=lr_finder)
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[1]['lr'] == 0.001

        lr_finder.min_lr = 0.01
        lr_finder.on_train_begin(self=lr_finder, logs={'test': 'sentinel'})
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[1]['lr'] == 0.01

    def test_on_train_end(self, monkeypatch):
        """Test on_train_end

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        fpath_lr_finder_history = '/tmp/history.csv'
        if os.path.exists(fpath_lr_finder_history):
            os.remove(fpath_lr_finder_history)

        lr_finder = MagicMock()
        lr_finder.on_train_end = LRFinder.on_train_end

        lr_finder.history = {}
        lr_finder.dirpath_results = '/tmp'
        lr_finder.n_total_iterations = 100
        lr_finder._plot_loss = MagicMock()
        lr_finder._plot_lr = MagicMock()

        plt_savefig = MagicMock()
        monkeypatch.setattr('ktorch.callbacks.plt.savefig', plt_savefig)

        lr_finder.on_train_end(self=lr_finder)

        assert os.path.exists(fpath_lr_finder_history)
        assert plt_savefig.call_count == 6

        lr_finder._plot_loss.assert_any_call(logscale=True)
        lr_finder._plot_loss.assert_any_call(logscale=False)
        lr_finder._plot_smoothed_loss.assert_any_call(logscale=True)
        lr_finder._plot_smoothed_loss.assert_any_call(logscale=False)


class TestTensorBoard():
    """Tests for TensorBoard"""

    @pytest.fixture(scope='class')
    def mock_logs(self):
        """mock_logs object fixture"""

        return {
            'name1': 1, 'name2': 2,
            'batch': 5, 'size': 32
        }

    def test_init(self, monkeypatch):
        """Test __init__

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_init = MagicMock()
        monkeypatch.setattr(
            'ktorch.callbacks.Callback.__init__', mock_init
        )

        mock_file_writer = MagicMock()
        mock_file_writer.return_value = None
        monkeypatch.setattr(
            'ktorch.callbacks.FileWriter.__init__', mock_file_writer
        )

        test_cases = [
            {},
            {'update_freq': 'epoch'},
            {'log_dir': 'training_jobs/test_job', 'update_freq': 'epoch'}
        ]

        for test_case in test_cases:
            tensorboard = TensorBoard(**test_case)

            if 'update_freq' in test_case:
                assert tensorboard.update_freq == test_case['update_freq']
            else:
                assert tensorboard.update_freq == 'batch'

            if 'log_dir' in test_case:
                mock_file_writer.assert_called_once_with(
                    log_dir=test_case['log_dir']
                )
            else:
                mock_file_writer.assert_called_once_with(
                    log_dir='./logs'
                )

            mock_init.assert_called_once_with()
            assert tensorboard.samples_seen == 0
            mock_init.reset_mock()
            mock_file_writer.reset_mock()

    def test_write_logs(self, mock_logs, monkeypatch):
        """Test _write_logs

        :param mock_logs: mock_logs object fixture
        :type mock_logs: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        # mock Summary.__init__ to control the return value and ensure that
        # `TensorBoard.writer.add_summary` is called with the return value
        mock_summary = MagicMock()
        mock_summary.return_value = 'mock_summary_return'
        monkeypatch.setattr(
            'ktorch.callbacks.Summary', mock_summary
        )

        mock_value = MagicMock()
        mock_value.return_value = 'mock_value_return'
        monkeypatch.setattr(
            'ktorch.callbacks.Summary.Value', mock_value
        )

        mock_logs = {
            'name1': 1, 'name2': 2,
            'batch': 5, 'size': 32
        }

        test_cases = [
            {'logs': mock_logs, 'index': 2},
            {'logs': mock_logs, 'index': 5}
        ]

        for test_case in test_cases:
            tensorboard = MagicMock()
            tensorboard.writer = MagicMock(spec=FileWriter)
            tensorboard._write_logs = TensorBoard._write_logs

            tensorboard._write_logs(self=tensorboard, **test_case)

            mock_value.assert_has_calls([
                call(tag='name1', simple_value=1),
                call(tag='name2', simple_value=2),
            ], any_order=True)
            mock_summary.assert_has_calls([
                call(value=['mock_value_return']),
                call(value=['mock_value_return'])
            ], any_order=True)
            tensorboard.writer.add_summary.assert_has_calls([
                call('mock_summary_return', test_case['index']),
            ])
            tensorboard.writer.flush.assert_called_once()

            mock_summary.reset_mock()
            mock_value.reset_mock()
            tensorboard.writer.reset_mock()

    def test_on_batch_end(self, mock_logs):
        """Test on_batch_end

        :param mock_logs: mock_logs object fixture
        :type mock_logs: dict
        """

        tensorboard = MagicMock()
        tensorboard.update_freq = 'batch'
        tensorboard.samples_seen = 0
        mock_write_logs = MagicMock()
        tensorboard._write_logs = mock_write_logs

        tensorboard.on_batch_end = TensorBoard.on_batch_end

        tensorboard.on_batch_end(
            self=tensorboard, batch=1, logs=mock_logs
        )
        assert tensorboard.samples_seen == 32

        tensorboard.on_batch_end(
            self=tensorboard, batch=2, logs=mock_logs
        )
        assert tensorboard.samples_seen == 64

        tensorboard.update_freq = 'epoch'
        tensorboard.on_batch_end(
            self=tensorboard, batch=0, logs=mock_logs
        )
        assert tensorboard.samples_seen == 64

        mock_write_logs.assert_has_calls([
            call(mock_logs, 32), call(mock_logs, 64)
        ])

    def test_on_epoch_end(self, mock_logs):
        """Test on_epoch_end

        :param mock_logs: mock_logs object fixture
        :type mock_logs: dict
        """

        tensorboard = MagicMock()
        tensorboard.update_freq = 'epoch'
        mock_write_logs = MagicMock()
        tensorboard._write_logs = mock_write_logs

        tensorboard.on_epoch_end = TensorBoard.on_epoch_end

        tensorboard.on_epoch_end(
            self=tensorboard, epoch=0, logs=mock_logs
        )
        assert mock_write_logs.call_count == 1

        tensorboard.on_epoch_end(
            self=tensorboard, epoch=1, logs=mock_logs
        )
        assert mock_write_logs.call_count == 2

        tensorboard.update_freq = 'batch'
        tensorboard.on_epoch_end(
            self=tensorboard, epoch=2, logs=mock_logs
        )
        assert mock_write_logs.call_count == 2

        mock_write_logs.assert_has_calls([
            call(mock_logs, 0), call(mock_logs, 1)
        ])
