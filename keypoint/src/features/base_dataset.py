import pandas as pd
from typing import Union
from pathlib import Path
from configs import DataConfig
from .utils import get_pose_transforms
from transformers import FeatureExtractionMixin
from .pose_dataset import PoseDataset
import random


class LocalDataset:
    def __init__(self, data: list):
        self.data = data

    def select_columns(self, columns: list):
        new_data = []
        for row in self.data:
            new_row = {col: row[col] for col in columns if col in row}
            new_data.append(new_row)
        return LocalDataset(new_data)

    def shuffle(self, seed=42):
        new_data = list(self.data)
        random.Random(seed).shuffle(new_data)
        return LocalDataset(new_data)

    def take(self, n: int):
        return LocalDataset(self.data[:n])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class LocalDatasetDict(dict):
    def select_columns(self, columns: list):
        return LocalDatasetDict({k: v.select_columns(columns) for k, v in self.items()})

    def shuffle(self, seed=42):
        return LocalDatasetDict({k: v.shuffle(seed) for k, v in self.items()})




class BaseDataset:
    def __init__(
        self,
        data_config: DataConfig,
        **kwargs,
    ) -> None:
        self.data_config = data_config
        self.dataset, self.gloss2id, self.id2gloss = self._load()
        if self.data_config.debug:
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].take(10)

    def _load(self) -> tuple:
        dataset, gloss2id, id2gloss = self._load_from_local(
            data_dir=self.data_config.data_dir,
            subset=self.data_config.subset,
        )
        dataset = dataset.select_columns(
            ["video_id", "resolution", "gloss_id", "video", "pose"]
        )
        dataset = dataset.shuffle(seed=42)
        return dataset, gloss2id, id2gloss

    def _load_from_local(self, data_dir: str, subset: str = None) -> tuple:
        raise NotImplementedError

    def get_split(
        self,
        split: str,
        processor: FeatureExtractionMixin,
    ) -> PoseDataset:
        return self.__get_pose_split(split, processor)

    def __get_pose_split(
        self,
        split: str,
        processor: FeatureExtractionMixin,
    ) -> PoseDataset:
        transform = get_pose_transforms(split, processor, self.data_config.transform)
        return PoseDataset(
            dataset=self.dataset[split],
            transforms=transform,
        )
