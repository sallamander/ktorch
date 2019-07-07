"""Trains a simple deep NN on the MNIST dataset.

Gets to ~98.20% test accuracy after 20 epochs.
"""

from keras.datasets import mnist
import torch
from torch.nn import Dropout, Linear, ReLU, Sequential

from metrics import categorical_accuracy
from model import Model

BATCH_SIZE = 128
# change to None to run on the CPU, 'cuda:1' to run on GPU 1, etc.
DEVICE = 'cuda:0'
EPOCHS = 20
NUM_CLASSES = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

network = Sequential(
    Linear(in_features=784, out_features=512),
    ReLU(),
    Dropout(0.2),
    Linear(in_features=512, out_features=512),
    ReLU(),
    Dropout(0.2),
    Linear(in_features=512, out_features=NUM_CLASSES),
)
model = Model(network, device=DEVICE)

model.compile(
    loss='CrossEntropyLoss', optimizer='Adam', metrics=[categorical_accuracy]
)

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    n_epochs=EPOCHS,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, BATCH_SIZE)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
