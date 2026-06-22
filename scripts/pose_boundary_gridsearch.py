from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qipedc_video_preprocess.boundary_comparison import load_boundary_cases
from qipedc_video_preprocess.config import PreprocessConfig
from scripts.pose_boundary_sweep import action_intervals, boundaries_from_intervals


def main() -> int:
    cfg = PreprocessConfig(project_root=Path("."))
    cases = [
        case
        for case in load_boundary_cases(cfg.resolve("Dataset/true_label"), cfg)
        if case.gt_boundaries
    ]
    cache_dir = cfg.resolve("Dataset/logs/pose_feature_cache")
    features = {
        case.base_id: {key: value for key, value in np.load(cache_dir / f"{case.base_id}.npz").items()}
        for case in cases
    }
    gt_total = sum(len(case.gt_boundaries) for case in cases)
    fps_by_id = {case.base_id: float(case.source_props.fps) for case in cases}

    angles = [130, 135, 140, 145, 150, 155, 160, 165, 170]
    vises = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    mins = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    delays = [0, 2, 4, 6, 8, 10, 12]
    merges = [4, 6, 8, 10, 12, 15, 20, 25, 30]
    alphas = [i / 40 for i in range(41)]
    offsets = list(range(-20, 21))

    best_key = None
    best_payload = None
    count = 0
    for angle in angles:
        for vis in vises:
            for up in mins:
                for down in mins:
                    for delay in delays:
                        for merge in merges:
                            interval_by_id = {
                                case.base_id: action_intervals(
                                    features[case.base_id],
                                    angle=angle,
                                    visibility=vis,
                                    min_up=up,
                                    min_down=down,
                                    delay=delay,
                                    merge_gap=merge,
                                )
                                for case in cases
                            }
                            for alpha in alphas:
                                base_pred_by_id = {
                                    case.base_id: boundaries_from_intervals(
                                        interval_by_id[case.base_id], alpha=alpha, offset=0
                                    )[: len(case.gt_boundaries)]
                                    for case in cases
                                }
                                for offset in offsets:
                                    count += 1
                                    rows = []
                                    errors = []
                                    found = missed = extra = exact = within = 0
                                    for case in cases:
                                        gt = tuple(int(v) for v in case.gt_boundaries)
                                        pred = tuple(
                                            max(0, min(int(case.source_props.num_frames) - 1, value + offset))
                                            for value in base_pred_by_id[case.base_id]
                                        )
                                        matched = min(len(gt), len(pred))
                                        found += matched
                                        missed += max(0, len(gt) - len(pred))
                                        extra += max(0, len(pred) - len(gt))
                                        err = tuple(abs(pred[i] - gt[i]) for i in range(matched))
                                        errors.extend(err)
                                        exact += sum(value == 0 for value in err)
                                        within += sum(value <= round(0.3 * fps_by_id[case.base_id]) for value in err)
                                        rows.append(
                                            (case.base_id, gt, pred, err, tuple(interval_by_id[case.base_id]))
                                        )
                                    mae = float(np.mean(errors)) if errors else 9999.0
                                    med = float(np.median(errors)) if errors else 9999.0
                                    key = (within, exact, found, -missed - extra, -mae, -med)
                                    if best_key is None or key > best_key:
                                        best_key = key
                                        best_payload = (
                                            angle,
                                            vis,
                                            up,
                                            down,
                                            delay,
                                            merge,
                                            alpha,
                                            offset,
                                            found,
                                            missed,
                                            extra,
                                            exact,
                                            within,
                                            mae,
                                            med,
                                            rows,
                                        )
                                        print(
                                            "best",
                                            count,
                                            {
                                                "angle": angle,
                                                "vis": vis,
                                                "up": up,
                                                "down": down,
                                                "delay": delay,
                                                "merge": merge,
                                                "alpha": alpha,
                                                "offset": offset,
                                                "found": found,
                                                "missed": missed,
                                                "extra": extra,
                                                "exact": exact,
                                                "within": within,
                                                "gt": gt_total,
                                                "mae": mae,
                                                "med": med,
                                            },
                                            flush=True,
                                        )
                                        if within == gt_total and exact == gt_total and missed == 0 and extra == 0:
                                            print("PERFECT", flush=True)
                                            for row in rows:
                                                print(row[:4], flush=True)
                                            return 0

    print("DONE", best_payload[:15] if best_payload else None)
    if best_payload:
        for row in best_payload[15]:
            print(row[:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
