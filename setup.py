"""Setup file"""

from setuptools import find_packages, setup

setup(
    name='ktorch',
    version='0.5.1',
    description=(
        'A repository providing a Keras-like interface for training and '
        'predicting with PyTorch networks.'
    ),
    long_description=(
        'KTorch aims to provide a high level set of APIs for model training '
        'and evaluation using PyTorch networks. Its goal is to provide as '
        'similar of an API as possible to Keras Model class, including the '
        'ease of specifying metrics, callbacks, etc. to track during '
        'training.'
    ),
    author='Sean Sall',
    author_email='ssall@alumni.nd.edu',
    url="https://github.com/sallamander/ktorch",
    download_url=(
        "https://github.com/sallamander/ktorch/archive/v0.5.1-alpha.tar.gz"
    ),
    license='MIT',
    install_requires=[
        'keras>=2.2.4',
        'tensorboard>=1.14',
        'future>=0.17.1'
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-pep8'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages()
)
