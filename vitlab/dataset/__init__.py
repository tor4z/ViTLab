from typing import Tuple
from cfg import Opts
from mlutils import mod
from torch.utils.data.dataset import Dataset
from .bird import *
from .dog import *
from .image_net_mini import *


def get_dataset(
    opt: Opts
) -> Tuple[Dataset, Dataset]:
    dataset_cls = mod.get('dataset', opt.dataset)
    training_set, val_set = dataset_cls.get_train_val_set(opt)
    training_dataset = dataset_cls(
        opt, training_set, training=True, testting=False
    )
    val_dataset = dataset_cls(
        opt, val_set, training=False, testting=False
    )

    yield training_dataset, val_dataset


def get_test_dataset(
    opt: Opts
) -> Tuple[Dataset, Dataset]:
    dataset_cls = mod.get('dataset', opt.dataset)
    test_set = dataset_cls.get_test_set(opt)
    testting_dataset = dataset_cls(
        opt, test_set, training=False, testting=True
    )

    return testting_dataset
