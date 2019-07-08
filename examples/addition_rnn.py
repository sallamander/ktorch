# -*- coding: utf-8 -*-
"""Sequence to sequence learning for performing addition

Reference Implementation:
    https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py

Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be reversed, shown to increase performance in many tasks.
References:
    - "Learning to Execute": http://arxiv.org/abs/1410.4615
    - "Sequence to Sequence Learning with Neural Networks":
        http://papers.nips.cc/paper/
            5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target

Two digits reversed:
- One layer LSTM (128 hidden units), 5k training examples = ~98% train/test
  accuracy in ~127 epochs

Three digits reversed:
- One layer LSTM (128 hidden units), 50k training examples = ~99% train/test
  accuracy in ~34 epochs

Four digits reversed:
- One layer LSTM (128 hidden units), 400k training examples = ~99% train/test
  accuracy in ~6 epochs

Five digits reversed:
- One layer LSTM (128 hidden units), 550k training examples = ~99% train/test
  accuracy in ~6 epochs
"""

import numpy as np
import torch
from torch.nn import Linear, Module, ModuleList

from ktorch.metrics import categorical_accuracy
from ktorch.model import Model

# === Parameters for the model === #
DEVICE = 'cuda:0'  # change to None for CPU, 'cuda:1' for GPU 1, etc.
N_EPOCHS = 127
# Try replacing LSTM with GRU or RNN
RNN = torch.nn.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

# === Parameters for the dataset === #
DIGITS = 2
REVERSE = True
TRAINING_SIZE = 5000
# Maximum length of input is 'int + int' (e.g., '345+678'), where the maximum
# length of int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS
# All the numbers, plus sign and space for padding.
CHARS = '0123456789+ '


def generate_random_number():
    """Generate a random number for an addition problem

    :return: random number with number of digits <= DIGITS
    :rtype: int
    """

    return int(''.join(np.random.choice(list('0123456789'))
               for i in range(np.random.randint(1, DIGITS + 1))))


class CharacterTable(object):
    """Encodes and decodes true or predicted character sets"""

    def __init__(self, chars):
        """Init

        :param chars: characters that can appear in the input.
        :type chars: str
        """

        self.chars = sorted(set(chars))
        self.char_indices = dict(
            (char, index) for index, char in enumerate(self.chars)
        )
        self.indices_char = dict(
            (index, char) for index, char in enumerate(self.chars)
        )

    def encode(self, decoded_str, num_rows):
        """One-hot encode the provided string

        :param decoded_str: string to be encoded
        :type decoded_str: str
        :param num_rows: number of rows in the returned one-hot encoding; this
         is used to keep the number of rows for each data the same
        :type num_rows: int
        :return: array holding a one-hot encoding of the provided string
        :rtype: numpy.ndarray
        """

        encoded_array = np.zeros((num_rows, len(self.chars)))
        for idx_char, char in enumerate(decoded_str):
            encoded_array[idx_char, self.char_indices[char]] = 1
        return encoded_array

    def decode(self, encoded_array, calc_argmax=True):
        """Decode the given vector or 2D array to their character output

        :param encoded_array: a tensor of probabilities or one-hot
         representations; or a vector of character indices (used with
         `calc_argmax=False`)
        :type encoded_array: numpy.ndarray
        :param calc_argmax: if True, find the character index with maximum
         probability, defaults to `True`.
        :type calc_argmax: bool
        """

        if calc_argmax:
            encoded_array = encoded_array.argmax(dim=-1)

        decoded_array = ''.join(
            self.indices_char[val.tolist()] for val in encoded_array
        )
        return decoded_array


class Colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class SimpleRNN(Module):

    def __init__(self):
        """Init"""

        super().__init__()

        self.encoder = RNN(len(CHARS), HIDDEN_SIZE, batch_first=True)
        self.decoder_layers = ModuleList([
            RNN(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
            for _ in range(LAYERS)
        ])
        self.linear = Linear(in_features=HIDDEN_SIZE, out_features=len(CHARS))

    def forward(self, inputs):
        """Return the outputs from a forward pass of the network

        :param inputs: batch of input addition problems, of shape
         (BATCH_SIZE, MAXLEN, len(CHARS))
        :type inputs: torch.Tensor
        :return: outputs of SimpleRNN, of shape (BATCH_SIZE, DIGITS + 1)
        :rtype: torch.Tensor
        """

        layer, _ = self.encoder(inputs)
        layer = layer[:, -1:, :].repeat(1, DIGITS + 1, 1)
        for decoder_layer in self.decoder_layers:
            layer, _ = decoder_layer(layer)
        outputs = self.linear(layer)
        outputs = torch.transpose(outputs, 1, 2)

        return outputs


ctable = CharacterTable(CHARS)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    a, b = generate_random_number(), generate_random_number()
    # Skip any addition questions we've already seen, as well as
    # any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    # Pad the data with spaces such that it is always MAXLEN
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g. '12+345  ' becomes '  543+21'
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorizing questions and answers...')
x = np.zeros((len(questions), MAXLEN, len(CHARS)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(CHARS)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
y = torch.argmax(y, dim=2)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)


network = SimpleRNN()
model = Model(network, device=DEVICE)

model.compile(
    loss='CrossEntropyLoss',
    optimizer='Adam',
    metrics=[categorical_accuracy]
)

# Train the model each generation and show predictions against the validation
# dataset
for iteration in range(1, N_EPOCHS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              n_epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        rowx = rowx.to(DEVICE)
        preds = model.network.forward(rowx)
        _, preds = torch.max(preds, 1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0], calc_argmax=False)
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(Colors.ok + '☑' + Colors.close, end=' ')
        else:
            print(Colors.fail + '☒' + Colors.close, end=' ')
        print(guess)
