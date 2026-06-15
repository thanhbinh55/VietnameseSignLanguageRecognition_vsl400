from pathlib import Path
from typing import Tuple
from .base_dataset import BaseDataset, LocalDatasetDict, LocalDataset
from .visl_400 import load_visl_400


class VISL400Dataset(BaseDataset):
    def _load_from_local(
        self,
        data_dir: str,
        subset: str,
    ) -> Tuple[LocalDatasetDict, dict, dict]:
        data_dir = Path(data_dir)
        cams = subset.split("_")[1:]
        gloss2id_file = data_dir / "gloss.csv"

        data_dict = {}
        for cam in cams:
            data_dict[f"cam_{cam}"] = {
                "meta": data_dir / f"cam_{cam}.json",
                "data": data_dir,
            }

        train_df, val_df, test_df, gloss2id = load_visl_400(
            data_dict, 
            gloss2id_file if gloss2id_file.exists() else None
        )
        if self.data_config.modality == "pose":
            train_df = train_df[
                train_df["pose"].map(lambda path: Path(path).exists())
            ].reset_index(drop=True)
            val_df = val_df[
                val_df["pose"].map(lambda path: Path(path).exists())
            ].reset_index(drop=True)
            test_df = test_df[
                test_df["pose"].map(lambda path: Path(path).exists())
            ].reset_index(drop=True)
        id2gloss = {v: k for k, v in gloss2id.items()}

        dataset = LocalDatasetDict({
            "train": LocalDataset(train_df.to_dict('records')),
            "validation": LocalDataset(val_df.to_dict('records')),
            "test": LocalDataset(test_df.to_dict('records')),
        })

        return dataset, gloss2id, id2gloss
