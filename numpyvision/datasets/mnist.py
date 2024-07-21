import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np

from .utils import check_file_integrity, download_file, read_idx_file

TEMPORARY_DIR = os.path.join(tempfile.gettempdir(), "numpyvision")


class MNIST:
    """
    MNIST Dataset
    http://yann.lecun.com/exdb/mnist

    Attributes
    ----------
    train : bool, default=True
        If True, uses train split, otherwise uses test split.
    data : np.ndarray
        numpy array containing images from chosen split.
    targets : np.ndarray
        numpy array containing labels from chosen split.
    root : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    class_to_idx : dict[str, int]
        Mapping from class to indices

    Usage
    -----
    >>> from numpyvision.datasets import MNIST
    >>> mnist = MNIST(train=True)
    >>> type(mnist.data)
    <class 'numpy.ndarray'>
    >>> mnistdata.dtype
    dtype('uint8')
    >>> mnist.data.min()
    0
    >>> mnist.data.max()
    255
    >>> mnist.data.shape
    (60000, 28, 28)
    >>> mnist.targets.shape
    (60000,)

    Citation
    --------
    @article{lecun-98,
      author={Lecun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
      journal={Proceedings of the IEEE},
      title={Gradient-based learning applied to document recognition},
      year={1998},
      volume={86},
      number={11},
      pages={2278-2324},
      doi={10.1109/5.726791}
    }
    """

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    resources = {
        "train_images": (
            "train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        "train_labels": (
            "train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        "test_images": (
            "t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        "test_labels": (
            "t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    }

    def __init__(
        self,
        root: Optional[str] = None,
        train: bool = True,
        download: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        root : str, default='/tmp/<dataset_name>/'
            Directory where all files exist or will be downloaded to (if `download` is True).
        train : bool, default=True
            If True, uses train split, otherwise uses test split.
        download : bool, default=True
            If True and files don't exist in `root`, downloads all files to `root`.
        load : bool, default=True
            If True, loads data from files in `root`.
        verbose : bool, default=True
            If True, prints download logs.
        """

        self.train = train

        self.root = (
            os.path.join(TEMPORARY_DIR, type(self).__name__) if root is None else root
        )

        if download:
            self.download(verbose=verbose)

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        image : np.ndarray
        label : int
        """

        img = self.data[index]
        target = int(self.targets[index])
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def download(self, verbose: bool = True) -> None:
        """
        Download files from mirrors and save to `root`.

        Parameters
        ----------
        force : bool=False
            If True, downloads all files even if they exist.
        verbose : bool, default=True
            If True, prints download logs.
        """

        os.makedirs(self.root, exist_ok=True)

        for filename, md5 in self.resources.values():
            filepath = os.path.join(self.root, filename)

            if check_file_integrity(filepath, md5):
                continue

            download_file(self.mirrors, filename, filepath, verbose)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        split = "train" if self.train else "test"

        data = self._load_file(f"{split}_images")
        targets = self._load_file(f"{split}_labels")

        return data, targets

    def _load_file(self, key: str) -> np.ndarray:
        filename, md5 = self.resources[key]
        filepath = os.path.join(self.root, filename)

        if not check_file_integrity(filepath, md5):
            raise RuntimeError(
                f"Dataset '{key}' not found in '{filepath}' or MD5 "
                "checksum is not valid. "
                "Use download=True or .download() to download it"
            )

        return read_idx_file(filepath)


class FashionMNIST(MNIST):
    """
    Fashion-MNIST Dataset
    https://github.com/zalandoresearch/fashion-mnist

    Attributes
    ----------
    train : bool, default=True
        If True, uses train split, otherwise uses test split.
    data : np.ndarray
        numpy array containing images from chosen split.
    targets : np.ndarray
        numpy array containing labels from chosen split.
    root : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    class_to_idx : dict[str, int]
        Mapping from class to indices

    Usage
    -----
    >>> from numpyvision.datasets import FashionMNIST
    >>> fmnist = FashionMNIST()
    >>> fmnist.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @online{xiao2017/online,
      author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
      title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking
                      Machine Learning Algorithms},
      date         = {2017-08-28},
      year         = {2017},
      eprintclass  = {cs.LG},
      eprinttype   = {arXiv},
      eprint       = {cs.LG/1708.07747},
    }
    """

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    mirrors = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ]

    resources = {
        "train_images": (
            "train-images-idx3-ubyte.gz",
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        ),
        "train_labels": (
            "train-labels-idx1-ubyte.gz",
            "25c81989df183df01b3e8a0aad5dffbe",
        ),
        "test_images": (
            "t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554ea6380940ec79",
        ),
        "test_labels": (
            "t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
    }


class KMNIST(MNIST):
    """
    Kuzushiji-MNIST Dataset
    https://github.com/rois-codh/kmnist

    Attributes
    ----------
    train : bool, default=True
        If True, uses train split, otherwise uses test split.
    data : np.ndarray
        numpy array containing images from chosen split.
    targets : np.ndarray
        numpy array containing labels from chosen split.
    root : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    class_to_idx : dict[str, int]
        Mapping from class to indices

    Citation
    --------
    @online{clanuwat2018deep,
      author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto
                      and Alex Lamb and Kazuaki Yamamoto and David Ha},
      title        = {Deep Learning for Classical Japanese Literature},
      date         = {2018-12-03},
      year         = {2018},
      eprintclass  = {cs.CV},
      eprinttype   = {arXiv},
      eprint       = {cs.CV/1812.01718},
    }
    """

    classes = [
        "お - o",
        "き - ki",
        "す - su",
        "つ - tsu",
        "な - na",
        "は - ha",
        "ま - ma",
        "や - ya",
        "れ - re",
        "を - wo",
    ]

    mirrors = [
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    ]

    resources = {
        "train_images": (
            "train-images-idx3-ubyte.gz",
            "bdb82020997e1d708af4cf47b453dcf7",
        ),
        "train_labels": (
            "train-labels-idx1-ubyte.gz",
            "e144d726b3acfaa3e44228e80efcd344",
        ),
        "test_images": (
            "t10k-images-idx3-ubyte.gz",
            "5c965bf0a639b31b8f53240b1b52f4d7",
        ),
        "test_labels": (
            "t10k-labels-idx1-ubyte.gz",
            "7320c461ea6c1c855c0b718fb2a4b134",
        ),
    }


class K49(MNIST):
    """
    Kuzushiji-49 Dataset
    https://github.com/rois-codh/kmnist

    Attributes
    ----------
    train : bool, default=True
        If True, uses train split, otherwise uses test split.
    data : np.ndarray
        numpy array containing images from chosen split.
    targets : np.ndarray
        numpy array containing labels from chosen split.
    root : str
        Directory where all files exist or will be downloaded.
    classes : list[str]
        Class names.
    class_to_idx : dict[str, int]
        Mapping from class to indices

    Citation
    --------
    @online{clanuwat2018deep,
      author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto
                      and Alex Lamb and Kazuaki Yamamoto and David Ha},
      title        = {Deep Learning for Classical Japanese Literature},
      date         = {2018-12-03},
      year         = {2018},
      eprintclass  = {cs.CV},
      eprinttype   = {arXiv},
      eprint       = {cs.CV/1812.01718},
    }
    """

    classes = [
        "あ - a",
        "い - i",
        "う - u",
        "え - e",
        "お - o",
        "か - ka",
        "き - ki",
        "く - ku",
        "け - ke",
        "こ - ko",
        "さ - sa",
        "し - shi",
        "す - su",
        "せ - se",
        "そ - so",
        "た - ta",
        "ち - chi",
        "つ - tsu",
        "て - te",
        "と - to",
        "な - na",
        "に - ni",
        "ぬ - nu",
        "ね - ne",
        "の - no",
        "は - ha",
        "ひ - hi",
        "ふ - fu",
        "へ - he",
        "ほ - ho",
        "ま - ma",
        "み - mi",
        "む - mu",
        "め - me",
        "も - mo",
        "や - ya",
        "ゆ - yu",
        "よ - yo",
        "ら - ra",
        "り - ri",
        "る - ru",
        "れ - re",
        "ろ - ro",
        "わ - wa",
        "ゐ - i",
        "ゑ - e",
        "を - wo",
        "ん - n",
        "ゝ - iteration mark",
    ]

    mirrors = [
        "http://codh.rois.ac.jp/kmnist/dataset/k49/",
    ]

    resources = {
        "train_images": (
            "k49-train-imgs.npz",
            "7ac088b20481cf51dcd01ceaab89d821",
        ),
        "train_labels": (
            "k49-train-labels.npz",
            "44a8e1b893f81e63ff38d73cad420f7a",
        ),
        "test_images": (
            "k49-test-imgs.npz",
            "d352e201d846ce6b94f42c990966f374",
        ),
        "test_labels": (
            "k49-test-labels.npz",
            "4da6f7a62e67a832d5eb1bd85c5ee448",
        ),
    }

    def _load_file(self, key: str) -> np.ndarray:
        filename, md5 = self.resources[key]
        filepath = os.path.join(self.root, filename)

        if not check_file_integrity(filepath, md5):
            raise RuntimeError(
                f"Dataset '{key}' not found in '{filepath}' or MD5 "
                "checksum is not valid. "
                "Use download=True or .download() to download it"
            )

        return np.load(filepath)["arr_0"]
