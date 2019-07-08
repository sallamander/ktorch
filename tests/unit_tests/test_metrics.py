"""Unit tests for metrics"""

import numpy as np
import torch

from ktorch.metrics import binary_accuracy, categorical_accuracy


def test_binary_accuracy():
    """Test binary_accuracy"""

    test_cases = [
        {'y_true': [0, 1, 0, 0, 1],
         'y_pred': [0.6, 0.4, 0.25, 0.35, 0.6],
         'expected_accuracy': 0.6},
        {'y_true': [0, 1, 0, 0, 1],
         'y_pred': [0.6, 0.4, 0.5, 0.5, 0.4],
         'expected_accuracy': 0.4},
        {'y_true': [0, 1, 0, 0, 1],
         'y_pred': [0.6, 0.4, 0.6, 0.6, 0.4],
         'expected_accuracy': 0},
        {'y_true': [1, 0, 0, 0, 0],
         'y_pred': [0.6, 0.4, 0.4, 0.4, 0.4],
         'expected_accuracy': 1},
    ]

    for test_case in test_cases:
        y_true = np.array(test_case['y_true'])
        y_true = torch.from_numpy(y_true).float()

        y_pred = np.array(test_case['y_pred'])
        y_pred = torch.from_numpy(y_pred)

        accuracy = binary_accuracy(y_true, y_pred)
        assert np.isclose(
            accuracy, test_case['expected_accuracy'], atol=1e-4
        )


class TestCategoricalAccuracy(object):
    """Tests for categorical_accuracy"""

    def test_categorical_accuracy__seqlen_equals_1(self):
        """Test categorical_accuracy when sequence length is equal to 1

        This scenario is typical of image classification problems, where we
        only predict for a single item, resulting in a sequence length of one.

        Each test case contains a 'y_true' that is of shape (batch_size, ),
        and a 'y_pred' that is of shape (batch_size, num_classes). Since
        sequence_length is 1, this dimension is ommitted.
        """

        test_cases = [
            {'y_true': [0, 1, 2],
             'y_pred': [[0.2, 0.1, 0.1],
                        [0.1, 0.2, 0.1],
                        [0.1, 0.1, 0.2]],
             'expected_accuracy': 1.0},
            {'y_true': [0, 1, 2],
             'y_pred': [[0.1, 0.2, 0.1],
                        [0.2, 0.1, 0.1],
                        [0.1, 0.2, 0.1]],
             'expected_accuracy': 0.0},
            {'y_true': [0, 1, 2],
             'y_pred': [[0.2, 0.1, 0.1],
                        [0.1, 0.2, 0.1],
                        [0.1, 0.2, 0.1]],
             'expected_accuracy': 0.66666},
        ]

        for test_case in test_cases:
            y_true = np.array(test_case['y_true'])
            y_true = torch.from_numpy(y_true).long()

            y_pred = np.array(test_case['y_pred'])
            y_pred = torch.from_numpy(y_pred)

            accuracy = categorical_accuracy(y_true, y_pred)
            assert np.isclose(
                accuracy, test_case['expected_accuracy'], atol=1e-4
            )

    def test_categorical_accuracy__seqlen_greater_than_1(self):
        """Test categorical_accuracy when sequence length is greater than 1

        This is common in sequence to sequence learning, where we might predict
        a sequence of characters.

        Each test case contains a 'y_true' that is of shape (sequence_length),
        and a 'y_pred' that is of shape (num_classes, sequence_length). During
        testing a batch dimension of random size is inserted at the beginning
        so that 'y_true' is of final shape (batch_size, sequence_length) and
        'y_pred' is of final shape (batch_size, num_classes, sequence_length).
        """

        test_cases = [
            {'y_true': [0, 1, 2],
             'y_pred': [[0.2, 0.1, 0.1],
                        [0.1, 0.2, 0.1],
                        [0.1, 0.1, 0.2]],
             'expected_accuracy': 1.0},
            {'y_true': [0, 1, 2],
             'y_pred': [[0.1, 0.2, 0.1],
                        [0.2, 0.1, 0.2],
                        [0.1, 0.1, 0.1]],
             'expected_accuracy': 0.0},
            {'y_true': [0, 1, 2],
             'y_pred': [[0.2, 0.1, 0.1],
                        [0.1, 0.2, 0.2],
                        [0.1, 0.1, 0.1]],
             'expected_accuracy': 0.66666},
        ]

        for test_case in test_cases:
            batch_size = np.random.randint(1, 8, 1, dtype=np.int8)[0]
            batch_axis = 0

            y_true = np.array(test_case['y_true'])
            y_true = np.expand_dims(y_true, axis=batch_axis)
            y_true = np.concatenate([y_true] * batch_size, axis=batch_axis)
            y_true = torch.from_numpy(y_true).long()

            y_pred = np.array(test_case['y_pred'])
            y_pred = np.expand_dims(y_pred, axis=batch_axis)
            y_pred = np.concatenate([y_pred] * batch_size, axis=batch_axis)
            y_pred = torch.from_numpy(y_pred)

            accuracy = categorical_accuracy(y_true, y_pred)
            assert np.isclose(
                accuracy, test_case['expected_accuracy'], atol=1e-4
            )
