"""Metrics to use during training"""

import torch


def categorical_accuracy(y_true, y_pred):
    """Return the accuracy in a multi-class problem

    :param y_true: ground truth target values, of shape (batch_size, )
    :type y_true: torch.Tensor
    :param y_pred: predicted target values, of shape (batch_size, num_classes)
    :type y_pred: torch.Tensor
    :return: categorical accuracy
    :rtype: numpy.ndarray
    """

    return torch.sum(
        y_true == torch.argmax(y_pred, dim=-1)
    ).cpu().numpy() / y_true.shape[0]
