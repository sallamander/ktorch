"""Setup file

This doesn't yet list any dependencies (call those a TODO) - it exists solely
to allow for pip installs with the develop flag (e.g. `pip install -e ktorch`).
"""

from setuptools import setup

setup(
    name='ktorch',
    version='0.1.0',
    description=(
        'A repository providing a Keras-like interface for training and '
        'predicting with PyTorch networks.'
    ),
    author='Sean Sall',
    license='MIT',
    install_requires=[
        'keras>=2.2.4',
        'torchvision>=0.2.2'
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-pep8'
        ]
    }
)
