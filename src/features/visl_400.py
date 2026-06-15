# Copyright 2023 Thinh T. Duong
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def load_visl_400(
    data_dict: Dict[str, Dict[str, Path]],
    gloss2id_file: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:

    dfs = []
    for cam, file_dict in data_dict.items():
        metadata_file = file_dict["meta"]
        data_dir = file_dict["data"] / cam
        df = pd.read_json(
            metadata_file,
            encoding='utf-8',
            dtype={
                "video_id": "string",
                "signer_id": "string",
                "fps": "int",
                "resolution": "int",
                "length": "float",
                "gloss": "string",
            }
        )
        df["cam_id"] = cam[-1]
        df["video"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.mp4"))
        df["pose"] = df["video_id"].apply(lambda x: str(data_dir / f"{x}.pose"))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    if gloss2id_file is not None and Path(gloss2id_file).exists():
        gloss2id = pd.read_csv(
            gloss2id_file,
            delimiter=",",
            names=["id", "gloss"],
            index_col="gloss",
        )
        gloss2id = gloss2id.to_dict()["id"]
    else:
        unique_glosses = sorted(df["gloss"].unique())
        gloss2id = {gloss: idx for idx, gloss in enumerate(unique_glosses)}

    df["gloss_id"] = df["gloss"].map(gloss2id)

    common_signer_ids = {
        "020": ("1", "2", "3"),
        "014": ("2", "3", "1"),
        "015": ("3", "1", "2"),
    }
    val_unique_signer_ids = ["007"]
    test_unique_signer_ids = ["024"]
    val_test_common_signer_ids = ["009"]
    train_not_unique_signer_ids = (
        val_unique_signer_ids
        + test_unique_signer_ids
        + val_test_common_signer_ids
        + list(common_signer_ids.keys())
    )
    cam_ids = list(df["cam_id"].unique())

    val_test_df = df[df["signer_id"].isin(val_test_common_signer_ids)]
    val_df = (
        val_test_df
        .groupby(["gloss_id", "cam_id"], group_keys=False)
        .sample(frac=0.5, random_state=42)
    )
    test_df = val_test_df[~val_test_df.index.isin(val_df.index)]

    train_df = df[~df["signer_id"].isin(train_not_unique_signer_ids)]
    val_df = pd.concat(
        [
            df[df["signer_id"].isin(val_unique_signer_ids)],
            val_df,
        ],
        ignore_index=True,
    )
    test_df = pd.concat(
        [
            df[df["signer_id"].isin(test_unique_signer_ids)],
            test_df,
        ],
        ignore_index=True,
    )

    for signer_id, (train_cam, val_cam, test_cam) in common_signer_ids.items():
        if train_cam in cam_ids:
            train_df = pd.concat(
                [
                    df[(df["signer_id"] == signer_id) & (df["cam_id"] == train_cam)],
                    train_df,
                ],
                ignore_index=True,
            )
        if val_cam in cam_ids:
            val_df = pd.concat(
                [
                    df[(df["signer_id"] == signer_id) & (df["cam_id"] == val_cam)],
                    val_df,
                ],
                ignore_index=True,
            )
        if test_cam in cam_ids:
            test_df = pd.concat(
                [
                    df[(df["signer_id"] == signer_id) & (df["cam_id"] == test_cam)],
                    test_df,
                ],
                ignore_index=True,
            )

    return train_df, val_df, test_df, gloss2id
