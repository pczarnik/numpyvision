"""
numpyvision
===========

numpyvision is a drop-in replacement for torchvision.datasets with an easy access
to MNIST-like datasets in a numpy format:
  1. MNIST
  2. FashionMNIST, FMNIST
  3. KMNIST, KuzushijiMNIST
  4. EMNIST - splitted into balanced, byclass, bymerge, digits, letters
    and mnist
  5. Kuzushiji49, K49

Every train or test split of datasets contains two numpy arrays:
  1. ``data`` of size ``(n_train_samples, width, height)``
  2. ``targets`` of size ``(n_train_samples,)``

All arrays are of type ``uint8``.

Example usage
-------------

  >>> from numpyvision.datasets import MNIST
  >>> mnist = MNIST()
  >>> type(mnist.data)
  <class 'numpy.ndarray'>
  >>> mnist.data.dtype
  dtype('uint8')
  >>> mnist.data.min()
  0
  >>> mnist.data.max()
  255
  >>> mnist.data.shape
  (60000, 28, 28)
  >>> mnist.targets.shape
  (60000,)

"""

__version__ = "0.5.0"

from . import datasets
