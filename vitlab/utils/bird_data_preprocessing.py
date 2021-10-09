from typing import Any, List, Mapping
import os
import pickle
import pandas as pd
from cfg import Opts
from mlutils import Log
import glob
import numpy as np
import tqdm
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


def read_class_dict(
    path: str
) -> Mapping[str, int]:
    output = {}

    with open(path, 'r') as f:
        table = pd.read_csv(f)
    length = len(table)

    for index in range(length):
        item = table.iloc[index]
        output[item['class']] = {
            'class_name': item['class'],
            'class_id': item['class_index'],
        }
    return output


def read_data_meta_csv(
    opt: Opts,
    class_dict: Mapping[str, Mapping[str, Any]]
) -> List[Mapping[str, Any]]:
    output = []

    with open(opt.meta_csv, 'r') as f:
        table = pd.read_csv(f)
    length = len(table)

    for index in range(length):
        item = table.iloc[index]
        output.append({
            'path': os.path.join(opt.data_root, item['filepaths']),
            'label': class_dict[item['labels']]['class_id'],
            'label_name': item['labels'],
            'dataset': item['data set']
        })
    return output


def find_all_file(root_path: str) -> List[str]:
    output = []
    for file in glob.glob(
        os.path.join(root_path, 'train', '*', '*.jpg')
    ):
        output.append(file)

    for file in glob.glob(
        os.path.join(root_path, 'test', '*', '*.jpg')
    ):
        output.append(file)

    for file in glob.glob(
        os.path.join(root_path, 'valid', '*', '*.jpg')
    ):
        output.append(file)

    return output


def preprocessing(opt: Opts) -> None:
    class_dict = read_class_dict(opt.class_dict_csv)
    meta_data = read_data_meta_csv(opt, class_dict)

    save_pickle(opt.meta_path, meta_data)
    Log.info('Done.')


def data_stat(opt: Opts) -> None:
    all_flies = find_all_file(opt.data_root)
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
