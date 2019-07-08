"""Trains an LSTM model on the IMDB sentiment classification task.

**Notes**
- The dataset is actually too small for LSTM to be of any advantage compared to
  simpler, much faster methods such as TF-IDF + LogReg.
- RNNs are tricky. Choice of batch size is important, choice of loss and
  optimizer is critical, etc.  Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different from what
  you see with CNNs/MLPs/etc.
"""

from keras.preprocessing import sequence
from keras.datasets import imdb
import torch
from torch.nn import Embedding, Linear, LSTM, Module, Sequential, Sigmoid

from metrics import binary_accuracy
from model import Model

BATCH_SIZE = 4
# change to None to run on CPU, 'cuda:1' to run on GPU 1, etc.
DEVICE = 'cuda:0'
MAX_FEATURES = 20000
# cut texts after this number of words (among top max_features most common
# words)
MAXLEN = 80
N_EPOCHS = 5

print('Loading data...')
try:
    (x_train, y_train), (x_test, y_test) = (
        imdb.load_data(num_words=MAX_FEATURES)
    )
except ValueError as ve:
    if 'allow_pickle' in str(ve):
        msg = (
            'You have run into an error that requires a downgrade of numpy '
            'to 1.16.2 or a change in the Keras source code such that the '
            'np.load() call in the imdb.load_data function uses the argument '
            'allow_pickle=True. \n'
        )
        raise ValueError(msg)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train = torch.from_numpy(x_train).long()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).long()
y_test = torch.from_numpy(y_test).float()


class SimpleClassifier(Module):
    """SimpleClassifier for IMDB text"""

    def __init__(self):
        """Init"""

        super().__init__()

        self.embedding = Embedding(MAX_FEATURES, 128)
        self.lstm = LSTM(input_size=128, hidden_size=32)
        self.linear = Linear(in_features=32 * MAXLEN, out_features=1)

    def to(self, *args, **kwargs):
        """"""

        super().to(*args, **kwargs)
        self.embedding.to("cpu")

    def forward(self, inputs):
        """Return the outputs from a forward pass of the network

        :param inputs: batch of input textual sequences, of shape
         (batch_size, MAXLEN)
        :type inputs: torch.Tensor
        :return: outputs of SimpleClassifier, of shape (BATCH_SIZE, )
        :rtype: torch.Tensor
        """

        inputs = inputs.cpu()

        layer = self.embedding(inputs)
        layer = layer.cuda()

        layer, _ = self.lstm(layer)
        layer = layer.view(layer.size(0), -1)
        layer = self.linear(layer)
        layer = torch.squeeze(layer)
        outputs = torch.nn.functional.sigmoid(layer)

        return outputs

print('Build model...')
network = SimpleClassifier()
model = Model(network, device=DEVICE)

# try using different optimizers and different optimizer configs
model.compile(
    loss='BCELoss', optimizer='Adam', metrics=[binary_accuracy]
)

print('Train...')
model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    validation_data=(x_test, y_test)
)
loss, accuracy = model.evaluate(
    x_test, y_test, batch_size=BATCH_SIZE
)
print('Test score:', loss)
print('Test accuracy:', accuracy)
