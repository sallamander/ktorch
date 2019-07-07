"""Unit tests for metrics"""

import numpy as np
import torch

from metrics import categorical_accuracy


def test_categorical_accuracy():
    """Tests for categorical_accuracy"""

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
        assert np.isclose(accuracy, test_case['expected_accuracy'])
