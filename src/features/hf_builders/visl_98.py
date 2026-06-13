# Copyright 2023 Thinh T. Duong
import pandas as pd
from pathlib import Path


def load_visl_98(metadata_file: Path, gloss2id_file: Path, data_dir: Path) -> tuple:
    gloss2id = pd.read_csv(
        gloss2id_file,
        delimiter=",",
        names=["id", "gloss"],
        index_col="gloss",
    )
    gloss2id = gloss2id.to_dict()["id"]

    df = pd.read_json(
        metadata_file,
        encoding='utf-8',
        dtype={
            "video_id": "string",
            "signer_id": "string",
            "fps": "int",
            "resolution": "int",
            "num_of_frames": "int",
            "length": "float",
            "gloss": "string",
            "english_gloss": "string",
        }
    )
    df["gloss_id"] = df["gloss"].map(gloss2id)
    df["video"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.mp4"))
    df["pose"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.pose"))

    test_set_signer_ids = ["001", "009", "014"]
    train_df = df[~df["signer_id"].isin(test_set_signer_ids)]
    test_df = df[df["signer_id"].isin(test_set_signer_ids)]

    return train_df, test_df, gloss2id
