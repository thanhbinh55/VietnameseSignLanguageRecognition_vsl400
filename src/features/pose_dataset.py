import numpy as np
import torch
from typing import Any
from torchvision.transforms.v2 import Compose
from torch.utils.data import Dataset as TorchDataset


class PoseDataset(TorchDataset):
    def __init__(
        self,
        dataset: list,
        transforms: Compose,
    ) -> None:
        self.dataset = dataset
        self.transforms = transforms
        self.num_videos = len(dataset)

    def __getitem__(self, index) -> Any:
        sample = self.dataset[index]
        path = sample["pose"]
        if isinstance(path, str) and path.endswith(".npy"):
            data = torch.from_numpy(np.load(path)).float()
        else:
            data = self.transforms(path)
        label = int(sample["gloss_id"])
        return {"pose": data, "label": label}

    def __len__(self) -> int:
        return len(self.dataset)
