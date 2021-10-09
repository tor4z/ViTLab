from typing import Any, List, Mapping, Tuple
import os
from cfg import Opts
from cvutils import imread
from cvutils import transform as tf
from mlutils import mod

from .utils import load_pickle, dataset_split
from .base import BaseDataset


__all__ = ['ImageNetMiniDataset']


Set = List[Mapping[str, Any]]


@mod.register('dataset')
class ImageNetMiniDataset(BaseDataset):
    def __init__(
        self,
        opt: Opts,
        dataset: Set,
        training: bool=True,
        testting: bool=False
    ) -> None:
        super().__init__(opt, dataset, training, testting)
        self.set_transform(opt)

    def set_transform(
        self,
        opt: Opts
    ) -> None:
        if self.training:
            self.transformer = tf.Compose([
                tf.TransposeTorch(),
                tf.Normalize(),
                tf.Resize(opt.input_size),
                tf.ToTensor()
            ])
        else:
            self.transformer = tf.Compose([
                tf.TransposeTorch(),
                tf.Normalize(),
                tf.Resize(opt.input_size),
                tf.ToTensor()
            ])

    @classmethod
    def get_test_set(cls, opt: Opts) -> Set:
        all_set = load_pickle(opt.meta_path)
        return all_set['val_set']

    @classmethod
    def get_train_val_set(cls, opt: Opts) -> Tuple[Set, Set]:
        all_set = load_pickle(opt.meta_path)
        train_set = all_set['train_set']
        val_set = all_set['val_set']

        return train_set, val_set

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        while True:
            item = self.dataset[index]
            label = item['label']
            image_path = item['path']

            if not os.path.exists(image_path):
                index += 1
                index = index % len(self)
                continue

            image = imread(image_path)
            image = self.transformer(image)

            return image, label
