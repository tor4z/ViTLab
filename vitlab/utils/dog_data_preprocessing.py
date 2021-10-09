from typing import Any, List, Mapping
import os
import glob
import tqdm
import pickle
import scipy.io
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


def load_mat(
    path: str
) -> Mapping[str, Any]:
    mat = scipy.io.loadmat(path)
    return mat


def find_all_file(root_path: str) -> List[str]:
    output = []
    for file in glob.glob(
        os.path.join(root_path, 'images', '*', '*.jpg')
    ):
        output.append(file)

    return output


def make_dataset(
    mat: Mapping[str, Any],
    root_path: str
) -> List[Mapping[str, Any]]:
    output = []

    for image_name, label in zip(mat['file_list'], mat['labels']):
        output.append({
            'path': os.path.join(root_path, str(image_name[0][0])),
            'label': int(label[0])
        })
    return output


def preprocessing(opt: Opts) -> None:
    train_mat = load_mat(opt.train_mat)
    test_mat = load_mat(opt.test_mat)

    image_root = os.path.join(opt.data_root, 'images')
    train_set = make_dataset(train_mat, image_root)
    test_set = make_dataset(test_mat, image_root)

    meta_data = {
        'train_set': train_set,
        'test_set': test_set
    }
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
