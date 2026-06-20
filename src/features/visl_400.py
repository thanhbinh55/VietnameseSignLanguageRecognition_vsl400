# Copyright 2023 Thinh T. Duong
# Modified 2025 Nguyen Viet Thanh Binh — added keypoint_dir support for flexible
# .pose file path resolution and dynamic split loading from split_info.csv.
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple



def get_pose_path(video_id: str, cam: str, data_dir: Path, keypoint_dir: Path = None) -> str:
    """Resolve the .pose file path for a given video_id.

    Search order (first existing path wins):
      1. keypoint_dir / cam / {video_id}.pose  -- separate keypoint folder, per-camera subfolder
      2. keypoint_dir / {video_id}.pose         -- separate keypoint folder, flat layout
      3. data_dir / cam / {video_id}.pose       -- co-located with dataset, per-camera subfolder
      4. data_dir / {video_id}.pose             -- co-located with dataset, flat layout

    If no existing file is found, returns candidate 1 (with keypoint_dir) or
    candidate 3 (without keypoint_dir) as the expected output path so that
    extraction scripts know where to write.
    """
    candidates = []
    if keypoint_dir is not None:
        candidates += [
            keypoint_dir / cam / f"{video_id}.pose",
            keypoint_dir / f"{video_id}.pose",
        ]
    candidates += [
        data_dir / cam / f"{video_id}.pose",
        data_dir / f"{video_id}.pose",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])  # default expected output path


def load_visl_400(
    data_dict: Dict[str, Dict[str, Path]],
    gloss2id_file: Path = None,
    keypoint_dir: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:

    dfs = []
    for cam, file_dict in data_dict.items():
        metadata_file = file_dict["meta"]
        data_dir = file_dict["data"] / cam
        kp_dir = Path(keypoint_dir) if keypoint_dir is not None else None
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
        df["pose"] = df["video_id"].apply(
            lambda x: get_pose_path(x, cam, file_dict["data"], kp_dir)
        )
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

    split_info_file = gloss2id_file.parent / "split_info.csv" if gloss2id_file else None
    if split_info_file and split_info_file.exists():
        split_df = pd.read_csv(split_info_file)
        split_map = dict(zip(split_df["video_id"], split_df["split"]))
        df["split"] = df["video_id"].map(split_map)
        
        train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
        val_df = df[df["split"].isin(["val", "validation"])].copy().reset_index(drop=True)
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    else:
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
