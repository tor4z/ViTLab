from typing import List, Mapping, Any
from torch.utils.data import Dataset
from cfg import Opts


Set = List[Mapping[str, Any]]


class BaseDataset(Dataset):
    def __init__(
        self,
        opt: Opts,
        dataset: Set,
        training: bool=True,
        testting: bool=False
    ) -> None:
        super().__init__()
        self.opt = opt
        self.dataset = dataset
        self.training = training
        self.testting = testting
