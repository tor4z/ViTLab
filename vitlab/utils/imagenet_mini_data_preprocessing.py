from typing import Any, List, Mapping
import os
import glob
import tqdm
import pickle
from collections import defaultdict
import numpy as np
from cfg import Opts
from mlutils import Log
from cvutils import imread


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


def find_all_file(root_path: str) -> List[str]:
    output = []
    for file in glob.glob(
        os.path.join(root_path, '*', '*.JPEG')
    ):
        output.append(file)

    return output


def from_path_to_label_name(path: str) -> str:
    return path.split('/')[-2]


def make_class_dict(
    files: List[str]
) -> Mapping[str, int]:
    class_name_dict = defaultdict(lambda: 0)
    class_dict = {}

    for file_path in files:
        label_name = from_path_to_label_name(file_path)
        class_name_dict[label_name] += 1

    for i, label_name in enumerate(class_name_dict.keys()):
        class_dict[label_name] = i

    return class_dict


def make_dataset(
    file_list: List[str],
    class_dict: Mapping[str, int]
) -> List[Mapping[str, Any]]:
    output = []

    for file_path in file_list:
        label_name = from_path_to_label_name(file_path)
        output.append({
            'path': file_path,
            'label': int(class_dict[label_name])
        })
    return output


def preprocessing(opt: Opts) -> None:
    train_path = os.path.join(opt.data_root, 'train')
    val_path = os.path.join(opt.data_root, 'val')

    train_files = find_all_file(train_path)
    val_files = find_all_file(val_path)
    class_dict = make_class_dict(train_files)

    train_set = make_dataset(train_files, class_dict)
    val_set = make_dataset(val_files, class_dict)

    meta_data = {
        'train_set': train_set,
        'val_set': val_set,
        'class_dict': class_dict
    }
    save_pickle(opt.meta_path, meta_data)

    Log.info('Done.')


def data_stat(opt: Opts) -> None:
    train_path = os.path.join(opt.data_root, 'train')
    val_path = os.path.join(opt.data_root, 'val')

    train_files = find_all_file(train_path)
    val_files = find_all_file(val_path)

    all_flies = train_files + val_files
    Log.info(f'All data length {len(all_flies)}.')
    mean_list = []
    std_list = []

    for file_name in tqdm.tqdm(all_flies):
        image = imread(file_name)
        image = image.reshape(3, -1)
        mean_list.append(image.mean(1))
        std_list.append(image.std(1))

    mean = np.stack(mean_list)
    std = np.stack(std_list)

    Log.info(f'mean: {mean.mean(0)}')
    Log.info(f'std: {std.mean(0)}')

    Log.info('Done.')
