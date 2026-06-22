from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qipedc_video_preprocess.boundary_comparison import load_boundary_cases
from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.pose_boundary import (
    _PoseIdx,
    _close_backend,
    _frame_to_timestamp_ms,
    _joint_angle,
    _load_pose_backend,
)


@dataclass(frozen=True)
class SweepResult:
    name: str
    params: dict[str, float | int | str]
    cases: int
    gt: int
    found: int
    missed: int
    extra: int
    exact: int
    within_03s: int
    mae_frames: float | None
    median_frames: float | None
    rows: tuple[dict[str, object], ...]

    @property
    def exact_rate(self) -> float:
        return self.exact / self.gt if self.gt else 0.0

    @property
    def within_rate(self) -> float:
        return self.within_03s / self.gt if self.gt else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--true-label-dir", default="Dataset/true_label")
    parser.add_argument("--cache-dir", default="Dataset/logs/pose_feature_cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = PreprocessConfig(project_root=Path(args.project_root))
    cases = [
        case
        for case in load_boundary_cases(cfg.resolve(args.true_label_dir), cfg)
        if case.gt_boundaries
    ]
    cache_dir = cfg.resolve(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_by_id = {
        case.base_id: load_or_extract_features(case, cfg, cache_dir, args.rebuild_cache)
        for case in cases
    }

    results: list[SweepResult] = []
    angle_values = (140, 145, 150, 155, 160, 165, 170)
    visibility_values = (0.35, 0.45, 0.5, 0.55, 0.6, 0.65)
    min_values = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    delay_values = (0, 2, 4, 6, 8, 10, 12)
    alpha_values = tuple(i / 20.0 for i in range(0, 21))
    offset_values = tuple(range(-12, 13, 2))

    total = (
        len(angle_values)
        * len(visibility_values)
        * len(min_values)
        * len(min_values)
        * len(delay_values)
        * len(alpha_values)
        * len(offset_values)
    )
    seen = 0
    best_key: tuple[float, float, float, float, float] | None = None

    for angle in angle_values:
        for visibility in visibility_values:
            for min_up in min_values:
                for min_down in min_values:
                    interval_cache = {
                        case.base_id: action_intervals(
                            features_by_id[case.base_id],
                            angle=angle,
                            visibility=visibility,
                            min_up=min_up,
                            min_down=min_down,
                            delay=0,
                            merge_gap=8,
                        )
                        for case in cases
                    }
                    for delay in delay_values:
                        delayed_cache = {
                            video_id: [(max(0, s - delay), e + delay) for s, e in intervals]
                            for video_id, intervals in interval_cache.items()
                        }
                        for alpha in alpha_values:
                            for offset in offset_values:
                                seen += 1
                                result = score_setting(
                                    cases,
                                    delayed_cache,
                                    alpha=alpha,
                                    offset=offset,
                                    name="gap_weight",
                                    params={
                                        "angle": angle,
                                        "visibility": visibility,
                                        "min_up": min_up,
                                        "min_down": min_down,
                                        "delay": delay,
                                        "alpha": alpha,
                                        "offset": offset,
                                    },
                                )
                                key = ranking_key(result)
                                if best_key is None or key > best_key:
                                    best_key = key
                                    results.append(result)
                                    print(
                                        f"best {seen}/{total}: exact={result.exact}/{result.gt} "
                                        f"within={result.within_03s}/{result.gt} mae={result.mae_frames} "
                                        f"params={result.params}",
                                        flush=True,
                                    )

    best = sorted(results, key=ranking_key, reverse=True)[: args.top]
    print("\nTOP")
    for result in best:
        print(
            json.dumps(
                {
                    "name": result.name,
                    "params": result.params,
                    "gt": result.gt,
                    "found": result.found,
                    "missed": result.missed,
                    "extra": result.extra,
                    "exact": result.exact,
                    "within_03s": result.within_03s,
                    "mae_frames": result.mae_frames,
                    "median_frames": result.median_frames,
                },
                ensure_ascii=False,
            )
        )
        for row in result.rows:
            print("  " + json.dumps(row, ensure_ascii=False))
    return 0


def ranking_key(result: SweepResult) -> tuple[float, float, float, float, float]:
    mae = result.mae_frames if result.mae_frames is not None else 1_000_000.0
    return (
        result.within_rate,
        result.exact_rate,
        -float(result.missed + result.extra),
        -mae,
        -float(result.median_frames or 1_000_000.0),
    )


def score_setting(
    cases,
    intervals_by_id: dict[str, list[tuple[int, int]]],
    *,
    alpha: float,
    offset: int,
    name: str,
    params: dict[str, float | int | str],
) -> SweepResult:
    errors: list[int] = []
    rows: list[dict[str, object]] = []
    gt_total = 0
    found = 0
    missed = 0
    extra = 0
    exact = 0
    within = 0
    for case in cases:
        gt = tuple(int(v) for v in case.gt_boundaries)
        pred_all = boundaries_from_intervals(
            intervals_by_id[case.base_id], alpha=alpha, offset=offset
        )
        # Pose is only an action-boundary signal. The method count still comes
        # from OCR/labels, so extras from side-view actions are not method
        # boundaries here.
        pred = tuple(pred_all[: len(gt)])
        matched = min(len(gt), len(pred))
        case_errors = tuple(abs(pred[i] - gt[i]) for i in range(matched))
        gt_total += len(gt)
        found += matched
        missed += max(0, len(gt) - len(pred))
        extra += max(0, len(pred) - len(gt))
        exact += sum(1 for err in case_errors if err == 0)
        within += sum(1 for err in case_errors if err <= round(0.3 * case.source_props.fps))
        errors.extend(case_errors)
        rows.append(
            {
                "video_id": case.base_id,
                "gt": gt,
                "pred": pred,
                "all_pred": tuple(pred_all),
                "err": case_errors,
                "intervals": tuple(intervals_by_id[case.base_id]),
            }
        )
    return SweepResult(
        name=name,
        params=params,
        cases=len(cases),
        gt=gt_total,
        found=found,
        missed=missed,
        extra=extra,
        exact=exact,
        within_03s=within,
        mae_frames=float(np.mean(errors)) if errors else None,
        median_frames=float(np.median(errors)) if errors else None,
        rows=tuple(rows),
    )


def boundaries_from_intervals(
    intervals: list[tuple[int, int]], *, alpha: float, offset: int
) -> list[int]:
    out: list[int] = []
    for prev, current in zip(intervals, intervals[1:]):
        rest_start = prev[1] + 1
        rest_end = current[0]
        boundary = int(round(rest_start + (rest_end - rest_start) * alpha + offset))
        out.append(max(rest_start, min(rest_end, boundary)))
    return out


def action_intervals(
    features: dict[str, np.ndarray],
    *,
    angle: float,
    visibility: float,
    min_up: int,
    min_down: int,
    delay: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    left = hand_intervals(features["left_angle"], features["left_visibility"], angle, visibility, min_up, min_down, delay)
    right = hand_intervals(features["right_angle"], features["right_visibility"], angle, visibility, min_up, min_down, delay)
    return merge_intervals(sorted(left + right), merge_gap)


def hand_intervals(
    angles: np.ndarray,
    visibility_values: np.ndarray,
    angle_threshold: float,
    visibility_threshold: float,
    min_up: int,
    min_down: int,
    delay: int,
) -> list[tuple[int, int]]:
    status = "down"
    up_frames = 0
    down_frames = 0
    start_temp: int | None = None
    end_temp: int | None = None
    start: int | None = None
    intervals: list[tuple[int, int]] = []
    for frame_index, (angle, visibility) in enumerate(zip(angles, visibility_values)):
        active = float(angle) < angle_threshold and float(visibility) >= visibility_threshold
        if active and status == "down":
            if up_frames == 0:
                start_temp = max(0, frame_index - delay)
            up_frames += 1
            if up_frames >= min_up:
                status = "up"
                start = start_temp
                up_frames = 0
                start_temp = None
        elif not active and status == "down":
            up_frames = 0
            start_temp = None

        if not active and status == "up":
            if down_frames == 0:
                end_temp = frame_index + delay
            down_frames += 1
            if down_frames >= min_down:
                status = "down"
                if start is not None and end_temp is not None and end_temp > start:
                    intervals.append((start, min(len(angles) - 1, end_temp)))
                start = None
                down_frames = 0
                end_temp = None
        elif active and status == "up":
            down_frames = 0
            end_temp = None
    return intervals


def merge_intervals(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def load_or_extract_features(case, cfg: PreprocessConfig, cache_dir: Path, rebuild: bool):
    cache_path = cache_dir / f"{case.base_id}.npz"
    if cache_path.exists() and not rebuild:
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}

    pose_state = _load_pose_backend(cfg, logging.getLogger("pose_sweep"))
    if pose_state.pose is None:
        raise RuntimeError("Pose backend unavailable")
    cap = cv2.VideoCapture(str(case.source_path))
    left_angle: list[float] = []
    right_angle: list[float] = []
    left_visibility: list[float] = []
    right_visibility: list[float] = []
    left_x: list[float] = []
    left_y: list[float] = []
    right_x: list[float] = []
    right_y: list[float] = []
    try:
        fps = float(case.source_props.fps)
        frame_index = 0
        while True:
            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                break
            values = extract_frame_features(frame, pose_state.pose, frame_index, fps)
            left_angle.append(values["left_angle"])
            right_angle.append(values["right_angle"])
            left_visibility.append(values["left_visibility"])
            right_visibility.append(values["right_visibility"])
            left_x.append(values["left_x"])
            left_y.append(values["left_y"])
            right_x.append(values["right_x"])
            right_y.append(values["right_y"])
            frame_index += 1
    finally:
        cap.release()
        _close_backend(pose_state)

    data = {
        "left_angle": np.array(left_angle, dtype=np.float32),
        "right_angle": np.array(right_angle, dtype=np.float32),
        "left_visibility": np.array(left_visibility, dtype=np.float32),
        "right_visibility": np.array(right_visibility, dtype=np.float32),
        "left_x": np.array(left_x, dtype=np.float32),
        "left_y": np.array(left_y, dtype=np.float32),
        "right_x": np.array(right_x, dtype=np.float32),
        "right_y": np.array(right_y, dtype=np.float32),
    }
    np.savez_compressed(cache_path, **data)
    return data


def extract_frame_features(frame, pose, frame_index: int, fps: float) -> dict[str, float]:
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose.detect_for_video(image, _frame_to_timestamp_ms(frame_index, fps))
    landmarks = getattr(result, "pose_landmarks", None)
    if not landmarks:
        return rest_values()
    lm = landmarks[0]
    left_shoulder = lm[_PoseIdx.LEFT_SHOULDER]
    left_elbow = lm[_PoseIdx.LEFT_ELBOW]
    left_wrist = lm[_PoseIdx.LEFT_WRIST]
    right_shoulder = lm[_PoseIdx.RIGHT_SHOULDER]
    right_elbow = lm[_PoseIdx.RIGHT_ELBOW]
    right_wrist = lm[_PoseIdx.RIGHT_WRIST]
    return {
        "left_angle": _joint_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y),
        ),
        "right_angle": _joint_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y),
        ),
        "left_visibility": visibility(left_wrist),
        "right_visibility": visibility(right_wrist),
        "left_x": finite(left_wrist.x),
        "left_y": finite(left_wrist.y),
        "right_x": finite(right_wrist.x),
        "right_y": finite(right_wrist.y),
    }


def rest_values() -> dict[str, float]:
    return {
        "left_angle": 180.0,
        "right_angle": 180.0,
        "left_visibility": 0.0,
        "right_visibility": 0.0,
        "left_x": math.nan,
        "left_y": math.nan,
        "right_x": math.nan,
        "right_y": math.nan,
    }


def visibility(landmark) -> float:
    return min(float(getattr(landmark, "visibility", 0.0)), float(getattr(landmark, "presence", 0.0)))


def finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
