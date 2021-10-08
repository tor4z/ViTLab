from typing import Any, List, Mapping
import pandas as pd


def read_class_dict(
    path: str
) -> Mapping[str, int]:
    output = {}

    with open(path, 'r') as f:
        table = pd.read_csv(f)


def read_data_meta_csv(
    path: str
) -> List[Mapping[str, Any]]:
    pass


