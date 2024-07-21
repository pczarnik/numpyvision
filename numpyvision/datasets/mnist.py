import os
import tempfile
from typing import Dict, Optional, Tuple, Literal
import string
import numpy as np

from .utils import check_file_integrity, extract_from_zip, download_file, read_idx_file

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

    def _filename_with_md5(self, key: str) -> Tuple[str, str]:
        return self.resources[key]

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
        filename, md5 = self._filename_with_md5(key)
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


class EMNIST(MNIST):
    """
    EMNIST Dataset
    https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Attributes
    ----------
    split : str
        One of: "byclass", "bymerge", "balanced", "letters", "digits", "mnists"
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
    >>> from numpyvision.datasets import EMNIST
    >>> emnist = EMNIST()
    >>> letters = emnist.Letters()
    >>> letters.train_images().dtype
    dtype('uint8')

    Citation
    --------
    @article{cohen2017emnist,
      title={EMNIST: an extension of MNIST to handwritten letters},
      author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
      year={2017},
      eprint={1702.05373},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    """

    mirrors = [
        "https://biometrics.nist.gov/cs_links/EMNIST/",
    ]

    resources = {"gzip": ("gzip.zip", "58c8d27c78d21e728a6bc7b3cc06412e")}
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
    _merged_classes = {
        "c",
        "i",
        "j",
        "k",
        "l",
        "m",
        "o",
        "p",
        "s",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    }
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        "byclass": sorted(list(_all_classes)),
        "bymerge": sorted(list(_all_classes - _merged_classes)),
        "balanced": sorted(list(_all_classes - _merged_classes)),
        "letters": ["N/A"] + list(string.ascii_lowercase),
        "digits": list(string.digits),
        "mnist": list(string.digits),
    }

    split_resources = {
        "byclass": {
            "train_images": (
                "emnist-byclass-train-images-idx3-ubyte.gz",
                "712dda0bd6f00690f32236ae4325c377",
            ),
            "train_labels": (
                "emnist-byclass-train-labels-idx1-ubyte.gz",
                "ee299a3ee5faf5c31e9406763eae7e43",
            ),
            "test_images": (
                "emnist-byclass-test-images-idx3-ubyte.gz",
                "1435209e34070a9002867a9ab50160d7",
            ),
            "test_labels": (
                "emnist-byclass-test-labels-idx1-ubyte.gz",
                "7a0f934bd176c798ecba96b36fda6657",
            ),
        },
        "bymerge": {
            "train_images": (
                "emnist-bymerge-train-images-idx3-ubyte.gz",
                "4a792d4df261d7e1ba27979573bf53f3",
            ),
            "train_labels": (
                "emnist-bymerge-train-labels-idx1-ubyte.gz",
                "491be69ef99e1ab1f5b7f9ccc908bb26",
            ),
            "test_images": (
                "emnist-bymerge-test-images-idx3-ubyte.gz",
                "8eb5d34c91f1759a55831c37ec2a283f",
            ),
            "test_labels": (
                "emnist-bymerge-test-labels-idx1-ubyte.gz",
                "c13f4cd5211cdba1b8fa992dae2be992",
            ),
        },
        "balanced": {
            "train_images": (
                "emnist-balanced-train-images-idx3-ubyte.gz",
                "4041b0d6f15785d3fa35263901b5496b",
            ),
            "train_labels": (
                "emnist-balanced-train-labels-idx1-ubyte.gz",
                "7a35cc7b2b7ee7671eddf028570fbd20",
            ),
            "test_images": (
                "emnist-balanced-test-images-idx3-ubyte.gz",
                "6818d20fe2ce1880476f747bbc80b22b",
            ),
            "test_labels": (
                "emnist-balanced-test-labels-idx1-ubyte.gz",
                "acd3694070dcbf620e36670519d4b32f",
            ),
        },
        "letters": {
            "train_images": (
                "emnist-letters-train-images-idx3-ubyte.gz",
                "8795078f199c478165fe18db82625747",
            ),
            "train_labels": (
                "emnist-letters-train-labels-idx1-ubyte.gz",
                "c16de4f1848ddcdddd39ab65d2a7be52",
            ),
            "test_images": (
                "emnist-letters-test-images-idx3-ubyte.gz",
                "382093a19703f68edac6d01b8dfdfcad",
            ),
            "test_labels": (
                "emnist-letters-test-labels-idx1-ubyte.gz",
                "d4108920cd86601ec7689a97f2de7f59",
            ),
        },
        "digits": {
            "train_images": (
                "emnist-digits-train-images-idx3-ubyte.gz",
                "d2662ecdc47895a6bbfce25de9e9a677",
            ),
            "train_labels": (
                "emnist-digits-train-labels-idx1-ubyte.gz",
                "2223fcfee618ac9c89ef20b6e48bcf9e",
            ),
            "test_images": (
                "emnist-digits-test-images-idx3-ubyte.gz",
                "a159b8b3bd6ab4ed4793c1cb71a2f5cc",
            ),
            "test_labels": (
                "emnist-digits-test-labels-idx1-ubyte.gz",
                "8afde66ea51d865689083ba6bb779fac",
            ),
        },
        "mnist": {
            "train_images": (
                "emnist-mnist-train-images-idx3-ubyte.gz",
                "3663598a39195d030895b6304abb5065",
            ),
            "train_labels": (
                "emnist-mnist-train-labels-idx1-ubyte.gz",
                "6c092f03c9bb63e678f80f8bc605fe37",
            ),
            "test_images": (
                "emnist-mnist-test-images-idx3-ubyte.gz",
                "fb51b6430fc4dd67deaada1bf25d4524",
            ),
            "test_labels": (
                "emnist-mnist-test-labels-idx1-ubyte.gz",
                "ae7f6be798a9a5d5f2bd32e078a402dd",
            ),
        },
    }

    def __init__(
        self,
        split: Literal["byclass", "bymerge", "balanced", "letters", "digits", "mnists"],
        **kwargs,
    ) -> None:
        if split not in self.splits:
            raise RuntimeError(
                f"Incorrect split {split}. Available splits: {self.splits}"
            )
        self.split = split

        super().__init__(**kwargs)
        self.classes = self.classes_split_dict[self.split]

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self._unzip_split()
        data, targets = super()._load_data()
        data = np.moveaxis(data, -2, -1)
        return data, targets

    def _filename_with_md5(self, key: str) -> Tuple[str, str]:
        return self.split_resources[self.split][key]

    def _unzip_split(self) -> None:
        """
        Unzip files from `zip_filepath` to `root`.

        Parameters
        ----------
        force : bool=False
            If True, unzips all files even if they exist.
        """

        os.makedirs(self.root, exist_ok=True)

        zip_filename, zip_md5 = self.resources["gzip"]
        zip_filepath = os.path.join(self.root, zip_filename)

        if not check_file_integrity(zip_filepath, zip_md5):
            raise RuntimeError(
                f"Zip file '{zip_filepath}' doesn't exists or its MD5"
                "checksum is not valid. "
                "Use EMNIST(download=True) or emnist.download() to download it"
            )

        for filename, md5 in self.split_resources[self.split].values():
            filepath = os.path.join(self.root, filename)

            if check_file_integrity(filepath, md5):
                continue

            extract_from_zip(zip_filepath, filename, self.root)
