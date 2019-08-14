"""Metrics to use during training"""

import torch


def binary_accuracy(y_true, y_pred):
    """Return the accuracy for a two-class problem

    Note that because this thresholds y_pred at 0.5, it assumes that y_pred
    represents predicted probabilities.

    :param y_true: ground truth target values, of shape (batch_size, )
    :type y_true: torch.Tensor with dtype float
    :param y_pred: predicted target values, of shape (batch_size, )
    :type y_pred: torch.Tensor
    :return: binary accuracy
    :rtype: numpy.ndarray
    """

    return torch.mean(
        (y_true == (y_pred > 0.5).float()).float()
    ).cpu().numpy()


def categorical_accuracy(y_true, y_pred):
    """Return the accuracy in a multi-class (3+ class) problem

    Note that because this simply takes the argmax of y_pred over the channel
    dimension (rather than thresholding it), it's not necessary that y_pred
    represent probabilities.

    :param y_true: ground truth target values, of shape
     (batch_size, sequence_length); sequence_length dimension can be ommitted
     if 1
    :type y_true: torch.Tensor with dtype long
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


class TopKCategoricalAccuracy(object):
    """Calculate the Top-K categorical accuracy"""

    def __init__(self, k=5):
        """Init

        :param k: number of elements to consider when calculating categorical
         accuracy
        :type k: int
        """

        self.k = k
        self.name = 'top_{}_categorical_accuracy'.format(k)

    def __call__(self, y_true, y_pred):
        """Calculate the categorical accuracy

        :param y_true: ground truth classifications, of shape (batch_size, )
        :type y_true: torch.Tensor
        ;param y_pred: predicted classifications of shape
         (batch_size, n_classes)
        :type y_pred: torch.Tensor
        :return: top-k categorical accuracy for the provided batch
        :rtype: float
        """

        top_k_classifications = torch.topk(y_pred, self.k)[1]
        n_correctly_classified = torch.sum(
            torch.eq(top_k_classifications, y_true.view(-1, 1))
        )
        n_correctly_classified = n_correctly_classified.float()

        return (n_correctly_classified / y_true.shape[0]).tolist()
