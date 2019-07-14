"""Tests for ktorch.callbacks"""

from unittest.mock import call, MagicMock

import pytest
from torch.utils.tensorboard.writer import FileWriter

from ktorch.callbacks import TensorBoard


class TestTensorBoard(object):
    """Test for TensorBoard"""

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
                    logdir=test_case['log_dir']
                )
            else:
                mock_file_writer.assert_called_once_with(
                    logdir='./logs'
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
