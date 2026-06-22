from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qipedc_video_preprocess.boundary_calibration import (
    action_intervals_from_pose_features,
    boundary_candidate_frames,
    choose_oracle_candidate,
    motion_signal,
)
from qipedc_video_preprocess.boundary_comparison import load_boundary_cases
from qipedc_video_preprocess.config import PreprocessConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--true-label-dir", default="Dataset/true_label")
    parser.add_argument("--cache-dir", default="Dataset/logs/pose_feature_cache")
    parser.add_argument("--output", default="Dataset/logs/pose_boundary_calibration.json")
    args = parser.parse_args()

    cfg = PreprocessConfig(project_root=Path(args.project_root))
    cases = [
        case
        for case in load_boundary_cases(cfg.resolve(args.true_label_dir), cfg)
        if case.gt_boundaries
    ]
    cache_dir = cfg.resolve(args.cache_dir)
    features = {
        case.base_id: {key: value for key, value in np.load(cache_dir / f"{case.base_id}.npz").items()}
        for case in cases
    }

    calibration = calibrate(cases, features, cfg)
    out_path = cfg.resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(calibration["summary"], ensure_ascii=False, indent=2))
    print(f"wrote={out_path}")
    return 0


def calibrate(cases, features, cfg: PreprocessConfig) -> dict[str, object]:
    # Candidate generation is deliberately broad. Pose supplies the action-rest
    # interval; hard-cut/motion supplies local peaks inside or near the rest gap.
    rows = []
    for case in cases:
        intervals = action_intervals_from_pose_features(
            features[case.base_id],
            cfg,
            merge_gap=8,
        )
        motion = motion_signal(case.source_path, int(case.source_props.num_frames))
        boundaries = []
        for idx, gt in enumerate(case.gt_boundaries):
            candidates = boundary_candidate_frames(
                intervals,
                motion,
                idx,
                source_frames=int(case.source_props.num_frames),
                cfg=cfg,
            )
            best = choose_oracle_candidate(candidates, int(gt))
            boundaries.append(
                {
                    "boundary_index": idx,
                    "gt": int(gt),
                    "chosen": best.frame,
                    "error": abs(best.frame - int(gt)),
                    "source": best.source,
                    "score": best.score,
                    "candidates": [
                        {"frame": c.frame, "source": c.source, "score": c.score}
                        for c in candidates
                    ],
                }
            )
        rows.append(
            {
                "video_id": case.base_id,
                "gt_boundaries": list(map(int, case.gt_boundaries)),
                "intervals": [list(map(int, interval)) for interval in intervals],
                "boundaries": boundaries,
            }
        )

    errors = [item["error"] for row in rows for item in row["boundaries"]]
    exact = sum(1 for err in errors if err == 0)
    within = sum(1 for err in errors if err <= 9)
    summary = {
        "cases": len(cases),
        "gt_boundaries": len(errors),
        "exact": exact,
        "within_03s_at_30fps": within,
        "mae_frames": float(np.mean(errors)) if errors else None,
        "median_frames": float(np.median(errors)) if errors else None,
        "max_error_frames": max(errors) if errors else None,
        "note": "oracle candidate selection over pose rest-gap + motion candidates; fitted on Dataset/true_label",
    }
    return {
        "summary": summary,
        "pose_config": {
            "pose_angle_threshold": cfg.pose_angle_threshold,
            "pose_visibility_threshold": cfg.pose_visibility_threshold,
            "pose_min_up_frames": cfg.pose_min_up_frames,
            "pose_min_down_frames": cfg.pose_min_down_frames,
            "pose_delay_frames": cfg.pose_delay_frames,
            "pose_boundary_gap_ratio": cfg.pose_boundary_gap_ratio,
            "pose_boundary_offset_frames": cfg.pose_boundary_offset_frames,
        },
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
