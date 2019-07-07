"""Trains a simple convnet on the MNIST dataset.

Gets to ~98.6%  test accuracy after 12 epochs.
"""

from keras.datasets import mnist
import torch
from torch.nn import (
    Conv2d, MaxPool2d, Dropout, Linear, Module, ReLU, Sequential
)

from metrics import categorical_accuracy
from model import Model

BATCH_SIZE = 128
# change to None to run on the CPU, 'cuda:1' to run on GPU 1, etc.
DEVICE = 'cuda:0'
EPOCHS = 12
NUM_CLASSES = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape so that channels are fist
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()


class SimpleCNN(Module):

    def __init__(self):
        """Init"""

        super().__init__()

        self.conv1 = Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )

        self.linear1 = Linear(in_features=(12 * 12 * 64), out_features=128)
        self.linear2 = Linear(in_features=128, out_features=NUM_CLASSES)

    def forward(self, inputs):
        """Return the outputs from a forward pass of the network

        :param inputs: batch of input images, of shape
         (BATCH_SIZE, n_channels, height, width)
        :type inputs: torch.Tensor
        :return: outputs of SimpleCNN, of shape (BATCH_SIZE, NUM_CLASSES)
        :rtype: torch.Tensor
        """

        layer = self.conv1(inputs)
        layer = ReLU()(layer)
        layer = self.conv2(layer)
        layer = ReLU()(layer)

        layer = MaxPool2d(kernel_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = layer.view(layer.size(0), -1)

        layer = self.linear1(layer)
        layer = Dropout(0.5)(layer)
        outputs = self.linear2(layer)

        return outputs

network = SimpleCNN()
model = Model(network, device=DEVICE)

model.compile(
    loss='CrossEntropyLoss',
    optimizer='Adam',
    metrics=[categorical_accuracy]
)

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          n_epochs=EPOCHS,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, BATCH_SIZE)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
