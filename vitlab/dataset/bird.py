from typing import Any, List, Mapping, Tuple
import os
from cfg import Opts
from cvutils import imread
from cvutils import transform as tf
from mlutils import mod

from .utils import load_pickle
from .base import BaseDataset


__all__ = ['BirdDataset']


Set = List[Mapping[str, Any]]


@mod.register('dataset')
class BirdDataset(BaseDataset):
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
                tf.ToTensor()
            ])
        else:
            self.transformer = tf.Compose([
                tf.TransposeTorch(),
                tf.Normalize(),
                tf.ToTensor()
            ])

    @classmethod
    def get_test_set(cls, opt: Opts) -> Set:
        output = []
        all_set = load_pickle(opt.meta_path)
        for item in all_set:
            if item['dataset'] == 'test':
                output.append(item)
        return output

    @classmethod
    def get_train_val_set(cls, opt: Opts) -> Tuple[Set, Set]:
        training_output = []
        val_output = []
        all_set = load_pickle(opt.meta_path)
        for item in all_set:
            if item['dataset'] == 'train':
                training_output.append(item)
            elif item['dataset'] == 'valid':
                val_output.append(item)
            else:
                # skip test data
                pass

        return training_output, val_output

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
