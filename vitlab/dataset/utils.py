from typing import Any, List, Mapping, Tuple
from collections import defaultdict
import pickle
import random


Set = List[Mapping[str, Any]]


def load_pickle(
    path: str
) -> Any:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(
    path: str, data: Any
) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def dataset_split(
    dataset: Set,
    training_proportion: float
) -> Tuple[Set, Set]:
    all_set_class = defaultdict(lambda: [])
    train_set = []
    val_set = []

    def _dataset_split(
        dataset: Set,
        training_proportion: float
    ) -> Tuple[Set, Set]:
        set_length = len(dataset)
        train_set_len = int(set_length * training_proportion)
        random.shuffle(dataset)
        train_set = dataset[:train_set_len]
        val_set = dataset[train_set_len:]
        return train_set, val_set

    for item in dataset:
        label = item['label']
        all_set_class[label].append(item)

    for _, curr_set in all_set_class.items():
        curr_train_sset, curr_val_sset = _dataset_split(
            curr_set, training_proportion
        )
        train_set += curr_train_sset
        val_set += curr_val_sset
    
    return train_set, val_set
