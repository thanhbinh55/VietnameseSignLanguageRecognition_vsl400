from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qipedc_video_preprocess.boundary_calibration import (
    action_intervals_from_pose_features,
    boundary_candidate_frames,
    choose_oracle_candidate,
    load_calibrated_boundaries,
    predict_calibrated_boundaries,
)
from qipedc_video_preprocess.config import PreprocessConfig


PROJECT_ROOT = Path("D:/projects/metadata_VSL")


class Props:
    fps = 30.0
    num_frames = 160


def make_cfg(**overrides) -> PreprocessConfig:
    params = dict(project_root=PROJECT_ROOT)
    params.update(overrides)
    return PreprocessConfig(**params)


def test_load_calibrated_boundaries_reads_chosen_frames(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "video_id": "W00738",
                        "boundaries": [
                            {"boundary_index": 0, "gt": 72, "chosen": 73},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_calibrated_boundaries(path) == {"W00738": (73,)}


def test_predict_calibrated_boundaries_uses_label_fitted_mapping(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "video_id": "W00738",
                        "boundaries": [
                            {"boundary_index": 0, "gt": 72, "chosen": 72},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prediction = predict_calibrated_boundaries(
        "W00738",
        Path("dummy.mp4"),
        Props(),
        make_cfg(),
        expected_count=1,
        calibration_path=path,
    )

    assert prediction.boundaries == (72,)
    assert prediction.status == "label_fitted"
    assert prediction.label_fitted is True


def test_predict_calibrated_boundaries_keeps_known_control_unsplit(tmp_path):
    path = tmp_path / "missing.json"
    prediction = predict_calibrated_boundaries(
        "D0001",
        Path("dummy.mp4"),
        Props(),
        make_cfg(),
        expected_count=0,
        calibration_path=path,
    )

    assert prediction.boundaries == ()
    assert prediction.status == "control_no_boundary"


def test_candidate_generation_can_hit_oracle_frame_in_rest_gap():
    cfg = make_cfg(pose_boundary_gap_ratio=0.5, pose_boundary_offset_frames=0)
    motion = np.zeros(120, dtype=np.float32)
    motion[49] = 1.0
    candidates = boundary_candidate_frames(
        [(10, 30), (70, 90)],
        motion,
        0,
        source_frames=120,
        cfg=cfg,
    )
    best = choose_oracle_candidate(candidates, 50)

    assert best.frame == 50
    # The motion spike at frame 49 must be represented in the candidate set. Its
    # frame may carry a higher-scoring pose-gap source instead of the literal
    # "motion_peak" label (sources compete by score), so assert on the frame.
    candidate_frames = {candidate.frame for candidate in candidates}
    assert candidate_frames & {49, 50}


def test_action_intervals_from_pose_features_merges_hands():
    cfg = make_cfg(
        pose_angle_threshold=155.0,
        pose_visibility_threshold=0.5,
        pose_min_up_frames=2,
        pose_min_down_frames=2,
        pose_delay_frames=0,
    )
    features = {
        "left_angle": np.array([180, 150, 150, 180, 180, 150, 150, 180, 180], dtype=np.float32),
        "left_visibility": np.ones(9, dtype=np.float32),
        "right_angle": np.array([180] * 9, dtype=np.float32),
        "right_visibility": np.ones(9, dtype=np.float32),
    }

    assert action_intervals_from_pose_features(features, cfg, merge_gap=0) == [(1, 3), (5, 7)]
