"""Metrics to use during training"""

import torch


def categorical_accuracy(y_true, y_pred):
    """Return the accuracy in a multi-class problem

    :param y_true: ground truth target values, of shape
     (batch_size, sequence_length); sequence_length dimension can be ommitted
     if 1
    :type y_true: torch.Tensor
    :param y_pred: predicted target values, of shape
     (batch_size, num_classes, sequence_length); sequence_length dimension can
     be ommited if 1
    :type y_pred: torch.Tensor
    :return: categorical accuracy
    :rtype: numpy.ndarray
    """

    return torch.mean(
        (y_true == torch.argmax(y_pred, dim=1)).float()
    ).cpu().numpy()
