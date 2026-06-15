import torch
from configs import DataConfig
from features import BaseDataset, VISL400Dataset


def load_dataset(data_config: DataConfig) -> BaseDataset:
    '''
    '''
    datasets = {
        "visl_400": VISL400Dataset,
    }
    return datasets[data_config.dataset](data_config)


def pose_collate_fn(examples) -> dict:
    # permute to (num_frames, num_channels, height, width)
    poses = torch.stack([example["pose"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return {"poses": poses, "labels": labels}
