# KTorch

KTorch aims to provide a high level set of APIs for model training and evaluation using PyTorch networks. Its goal is to provide as similar of an API as possible to Keras [`Model`](https://keras.io/models/model/) class, including the ease of specifying metrics, callbacks, etc. to track during training.

# Why KTorch?

There are a number of libraries out there that offer high level training / evaluation functionality for PyTorch networks, e.g. [`ignite`](https://github.com/pytorch/ignite), [`poutyne`](https://github.com/GRAAL-Research/poutyne), [`torchsample`](https://github.com/ncullen93/torchsample). KTorch offers a couple of important differences: 

- By modeling the KTorch API as closely as possible to the Keras API, users don't have to learn two sets of APIs to train networks with PyTorch versus a different Keras backend 
- Users are presented with the user-friendliness and intuitiveness of the Keras APIs for training / evaluation, which have proven to be incredibly easy to pick up, use, and iterate with
- If / when a PyTorch backend for Keras is implemented, users will be able to switch to using Keras almost seamlessly
- Direct use of Keras code where possible (e.g. the use of Keras `BaseLogger`, `CallbackList`, `History`, and `ProgbarLogger` in the `ktorch.model.Model` class)

# Getting Started: 30 seconds to KTorch

The core data structure of KTorch is the __Model__ class, modeled after the Keras [`Model`](https://keras.io/models/model/) class. It acts as a container for networks that are constructed using layers of [`torch.nn.Module`](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html) objects, and allows for easy training and evaluation of PyTorch networks.

First, construct a network (using either the `torch.nn.Sequential` class or by creating a subclass of `torch.nn.Module`):

```python
from torch.nn import Linear, ReLU, Sequential

network = Sequential(
    Linear(in_features=784, out_features=64),
    ReLU(),
    Linear(in_features=64, out_features=10),
)
```

Next, build a `Model` using that network:

```python
from ktorch import Model

model = Model(network)
```

Configure the learning process with `.compile()`:

```python
from ktorch.metrics import categorical_accuracy

model.compile(
    loss='CrossEntropyLoss',
    optimizer='Adam',
    metrics=[categorical_accuracy]
)
```

You can now train the model on batches of your training data:

```python
# x_train and y_train are torch.Tensor objects
model.fit(x_train, y_train, n_epochs=3, batch_size=32)
```

Alternatively you can feed batches to your model manually:

```python
model.train_on_batch(x_train, y_train)
```

You can also train using a generator that yields batches of your training data:

```python
model.fit_generator(generator, n_steps_per_epoch=512, n_epochs=3)
```

Evaluate your performance in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

In the [examples folder](https://github.com/sallamander/ktorch/tree/master/examples) of the repository, you'll find some more advanced models exemplifying KTorch functionality.

# Installation

Before installing KTorch, please [install PyTorch](https://pytorch.org/get-started/locally/). Then, install KTorch:

*Note*: These installation steps assume that you are on a Linux or Mac environment. If you're on Windows, you'll need to remove `sudo` from the command:

```sh
sudo pip install ktorch
```

# Configuring the KTorch backend

KTorch itself makes direct use of Keras functionality, which requires that you set a backend in the `~/.keras/keras.json` file. See the [Keras backend instructions](https://keras.io/backend/) for details on how to change backends. Since there is not yet a `pytorch` backend, KTorch comes packaged with a modified `numpy_backend` that allows for use of Keras functionality without having to install a different backend (e.g. `Tensorflow`). Simply use `"ktorch.numpy_backend"` in the "backend" key in your `~/.keras/keras.json`, and you should be all set!

# Support

This package is still in alpha mode. It started as (and largely still is) a collection of code used for my personal research, motivated by the desire for easy, extensible training and evaluation using PyTorch networks in the same way that Keras provides. It has been packaged together in the case that others would find it useful, and it will continue to be developed largely in line with the need for additional features / functionality in my personal research.

That being said, I would love it if the PyTorch community at large found this package useful, and from that perspective would be happy to develop additional features / functionality to make it as useful as possible (keeping in mind the guiding principle of providing a Keras-like API). Please submit **requests for new features**, as well as **bug reports** via [Github issues](https://github.com/sallamander/ktorch/issues).
