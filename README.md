# numpyvision: Vision datasets as numpy arrays

numpyvision is a drop-in replacement for torchvision.datasets with an easy access to MNIST and other MNIST-like datasets (FashionMNIST, KMNIST, EMNIST) in your numpy code.

numpyvision replicates the functionality of `torchvision.datasets.mnist` without the need to download dozens of dependencies - numpyvision has only one dependency: `numpy`.


## Usage

Each dataset stores train/test images as numpy arrays of shape `(n_samples, img_height, img_width)` and train/test labels as numpy arrays of shape `(n_samples,)`.

MNIST example:
```python
>>> from numpyvision.datasets import MNIST
>>> mnist = MNIST(train=True)
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
>>> mnist.classes[:3]
['0 - zero', '1 - one', '2 - two']
```

FashionMNIST example:
```python
from numpyvision.datasets import FashionMNIST
import matplotlib.pyplot as plt

fmnist = FashionMNIST()
img, label = fmnist[0]
plt.imshow(img, cmap='gray')
plt.title(fmnist.classes[label])
plt.axis('off')
plt.show()
```
![FashionMNIST example](https://raw.githubusercontent.com/pczarnik/numpyvision/main/imgs/fmnist_boot.png)

EMNIST example
```python
from numpyvision.datasets import EMNIST
import matplotlib.pyplot as plt

letters = EMNIST('letters')
plt.imshow(
    letters.data[:256]
        .reshape(16, 16, 28, 28)
        .swapaxes(1, 2)
        .reshape(16 * 28, -1),
    cmap='gray')
plt.axis('off')
plt.show()
```
![EMNIST example](https://raw.githubusercontent.com/pczarnik/numpyvision/main/imgs/emnist_letters_256.png)

## Installation

Install `numpyvision` from [PyPi](https://pypi.org/project/numpyvision):
```
pip install numpyvision
```
or from source:
```
pip install -U git+https://github.com/pczarnik/numpyvision
```
The only requirements for numpyvision are `numpy>=1.22` and `python>=3.9`.

If you want to have progress bars while downloading datasets, install with
```
pip install numpyvision[tqdm]
```

## Acknowledgments

The main inspirations for numpyvision were [`mnist`](https://github.com/datapythonista/mnist) and [`torchvision.datasets.mnist`](https://github.com/pytorch/vision).
